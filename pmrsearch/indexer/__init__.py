import os
import pandas as pd
import re
import rdflib
import rdflib.graph
import torch.backends
import torch.backends.mps
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import codecs
import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import logging as log
from lxml import etree
import re

from ..setup import RESOURCE_PATH, SEARCH_FILE, BERTModel, BIOBERT, NLPModel, METADATA, METADATA_FILE, WORKSPACE_DIR, url_to_curie, loadJson, getAllFilesInDir, dumpJson
from .sckan_crawler import extract_sckan_terms
from .clusterer import CellmlClusterer
from .ontology_crawler import load_ontologies

RS_WORKSPACE = f'{RESOURCE_PATH}/listOfWorkspace.json'

ALPHA = 0.5
BETA = 0.5
NLP_THRESHOLD = 0.9
SEARCH_THRESHOLD = 0.8
SEARCH_BIO_THRESHOLD = 0.8

def to_embedding(term_data, model, nlp_model):
    # get embedding of the main label
    doc = nlp_model(term_data['label'])
    ents = [ent.text for ent in doc.ents]
    ents = [doc.text] if len(ents) == 0 else ents
    embs = torch.mean(model.encode(ents, convert_to_tensor=True), 0)
    
    # add synonyms
    syn_embs = []
    for synonym in term_data['synonym']:
        doc = nlp_model(synonym)
        ents = [ent.text for ent in doc.ents]
        ents = [doc.text] if len(ents) == 0 else ents
        syn_embs += [torch.mean(model.encode(ents, convert_to_tensor=True), 0)]
    if len(syn_embs):
        syn_embs = torch.mean(torch.stack(syn_embs), 0) * ALPHA
        embs = (embs + syn_embs) / (1 + ALPHA)
    
    ## add parents
    added_embs = []
    for added_term in term_data['is_a_text']:
        doc = nlp_model(added_term)
        ents = [ent.text for ent in doc.ents]
        ents = [doc.text] if len(ents) == 0 else ents
        added_embs += [torch.mean(model.encode(ents, convert_to_tensor=True), 0)]
    if len(added_embs) > 0:
        added_embs = torch.mean(torch.stack(added_embs), 0) * BETA
        embs = (embs + added_embs) / (1 + BETA)
    return embs

class PMRIndexer:
    def __init__(self, pmr_workspace_dir, bert_model=None, biobert_model=None, nlp_model=None, sckan_version=None):
        device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        log.info(f'loading {BERTModel}')
        self.__bert_model = SentenceTransformer(BERTModel, device=device) if bert_model is None else bert_model
        log.info(f'loading {BIOBERT}')
        self.__biobert_model = SentenceTransformer(BIOBERT, device=device) if biobert_model is None else biobert_model
        self.__nlp_model = spacy.load(NLPModel) if nlp_model is None else nlp_model
        self.__nlp_model.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": NLP_THRESHOLD})
        self.__linker = self.__nlp_model.get_pipe("scispacy_linker")
        self.__nlp_model.add_pipe("abbreviation_detector")
        self.__pmr_workspace_dir = pmr_workspace_dir if pmr_workspace_dir is not None else WORKSPACE_DIR
        self.__sckan_version = sckan_version
        self.__ontologies = load_ontologies()

    def __sckan_search(self, query, model, sckan_embs, k=10, th=0.7):
        query_emb = model.encode(query, show_progress_bar=False,  convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, sckan_embs['embs'])[0]
        top_results = torch.topk(cos_scores, k=k)

        results = {}
        for score, idx in zip(top_results[0], top_results[1]):
            if score < th:
                break
            curie = sckan_embs['id'][idx.item()]
            if curie not in results:
                results [curie]= (self.__sckan_terms[curie]['label'], score.item())
            if len(results) == k:
                break
        return results

    def __get_sckan_candidate(self, query):
            r1 = self.__sckan_search(query, self.__bert_model, self.__sckan_bert_embs, k=10, th=SEARCH_THRESHOLD)
            r2 = self.__sckan_search(query, self.__biobert_model, self.__sckan_biobert_embs, k=10, th=SEARCH_BIO_THRESHOLD)
            for r2_ids in r2:
                if r2_ids in r1:
                    return (r2_ids, r2[r2_ids][0], (r2[r2_ids][1]+r1[r2_ids][1])/2)

    def __get_cellml_documentation(self, html_path):
        # pattern to handle abbreviations
        pattern = r'\((\w+)s\)'

        documentation = []
        try:
            file_name = codecs.open(html_path, "r", "utf-8")
            soup = BeautifulSoup(file_name.read(), 'lxml')
        except Exception:
            file_name = codecs.open(html_path, "r", "latin-1")
            soup = BeautifulSoup(file_name.read(), "lxml")
        
        if soup.title is not None:
            title = re.sub(pattern, r'(\1)', str(soup.title.text))
            if title != '':
                documentation += [title]
        
        paragraphs = soup.find_all('p') + soup.find_all('para')
        for p in paragraphs:
            if p is not None:
                try:
                    p = re.sub('\s+', ' ', p.text).strip()
                    p = re.sub(pattern, r'(\1)', p)
                    if len(p.split()) > 50 and len(p) > 250:
                        documentation += p.split('. ')
                except Exception:
                    log.warning(f'Cannot load a paragraph from {file_name}')
        return list(set(documentation))

    def __extract_pmr(self, clean_extraction):
        ### Parsing all cellml and rdf files from workspaces
        
        # load available search_data
        search_data = torch.load(SEARCH_FILE)
        if clean_extraction:
            search_data = {}

        # initialisation
        term_ids = []
        all_documentation = []
        file_types = ['cellml', 'rdf', 'html', 'md', 'rst']
        workspaces = loadJson(RS_WORKSPACE)
        
        cellml_to_terms = {}

        # filter workspaces based on clean_extraction
        if clean_extraction:
            selected_workspaces = workspaces
            cellmls = {}
        else:
            cellmls = search_data['cellml']
            selected_workspaces = {}
            for workspace_url, workspace in workspaces.items():
                if not workspace.get('hasExtracted'):
                    for cellml_url in set(workspace.get('cellml', [])):
                        del cellmls[cellml_url]
                    workspace['cellml'] = []
                    selected_workspaces[workspace_url] = workspace

        new_cellmls = []
        # sckan all selected workspaces
        for workspace_url, workspace in selected_workspaces.items():
            workspace_dir = os.path.join(self.__pmr_workspace_dir, workspace['workingDir'])
            all_files = [file for file in getAllFilesInDir(workspace_dir)
                        if any(file.endswith(ext) for ext in file_types)]
            g_rdf = rdflib.Graph()
            workspace_documentation = []
            for file_name in all_files:
                if (file_type:=file_name[file_name.rfind('.') + 1:].lower()) == 'cellml': ## if cellml file found
                    cellml_url = f'{workspace_url}/rawfile/{workspace["commit"]}/{file_name[len(workspace_dir) + 1:]}'
                    cellmls[cellml_url] = {'workspace': workspace_url, 'workingDir': workspace['workingDir'], 'rdfLeaves': []}
                    # extract from cellml xml
                    parser = etree.XMLParser(recover=True, remove_comments=True)
                    root = etree.parse(file_name, parser).getroot()
                    g = rdflib.Graph()
                    for rdfElement in root.xpath(".//*[local-name()='RDF']"):
                        try:
                            g.parse(data=etree.tostring(rdfElement), format="application/rdf+xml")
                        except:
                            log.error(f'Cannot parse RDF in {cellml_url}')
                    cellmls[cellml_url]['rdfLeaves'] += [url_to_curie(str(o)) for o in g.objects() if url_to_curie(str(o)) is not None]
                    print('new', cellml_url, cellmls[cellml_url]['rdfLeaves'])
                    term_ids += cellmls[cellml_url]['rdfLeaves']
                    # add exposure information
                    if 'exposures' in workspace:
                        cellmls[cellml_url]['exposure'] = list(workspace['exposures'].keys())[-1]
                    # add sha information
                    cellmls[cellml_url]['sha'] = workspace['commit']
                    # add relative cellml path
                    cellmls[cellml_url]['cellml'] = file_name[len(workspace_dir) + 1:]
                    # add cellml_url to its workspace
                    if 'cellml' not in workspace:
                        workspace['cellml'] = []
                    workspace['cellml'] += [cellml_url]
                    # update documentation
                    if len(documents:=self.__get_cellml_documentation(file_name)) > 0:
                        cellmls[cellml_url]['documentation'] = documents
                        all_documentation += documents
                    # add to new_cellml to be checked later
                    new_cellmls += [cellml_url]
                elif file_type == 'rdf': ## if rdf file found, add to cellml files
                    try:
                        g_rdf.parse(file_name)
                    except:
                        log.error(f'Cannot parse RDF in {file_name}')
                elif file_type in ['html', 'md', 'rst']:
                    if len(documents:=self.__get_cellml_documentation(file_name)) > 0:
                        workspace_documentation += documents
            
            for cellml_url in workspace.get('cellml', []):
                cellmls[cellml_url]['rdfLeaves'] = list(set(cellmls[cellml_url]['rdfLeaves'] + list(g_rdf.objects())))
                # extract documentation
                if len(cellml_doc:=cellmls[cellml_url].get('documentation', [])) > 0 or len(workspace_documentation) > 0:
                    cellmls[cellml_url]['documentation'] = list(set(cellml_doc + workspace_documentation))
                ## load cellml_to_terms
                cellml_to_terms[cellml_url] = cellmls[cellml_url]['rdfLeaves']

            term_ids += [url_to_curie(str(o)) for o in g_rdf.objects() if url_to_curie(str(o)) is not None]
            all_documentation += workspace_documentation
            
            # update workspace, stated hat it has been extracted
            workspace['hasExtracted'] = True

        ### Get term's label, synonym, etc for ontology type 
        pmr_terms = {}
        for term_id in set(term_ids):
            if term_id in self.__ontologies.index:
                term = {'label':'', 'def':'', 'synonym':[], 'is_a':[], 'is_a_text':[]}
                onto = self.__ontologies.loc[term_id]
                term['label'] = onto['name']
                definition = onto['def'] if pd.notna(onto['def']) else ''
                term['def'] = re.findall(r'"([^"]*)"', definition) if len(definition) > 0 else []
                syn = onto['synonym'] if pd.notna(onto['synonym']) else ''
                term['synonym'] = list(set(re.findall(r'"([^"]*)"', syn) if len(syn) > 0 else []))
                if not pd.isna(onto['is_a']):
                    substrs = onto['is_a'].split('|')
                    term['is_a'] = [s.split(' ! ')[0] for s in substrs]
                    term['is_a_text'] = list(set([self.__ontologies.loc[t]['name'] for t in term['is_a']]))
                pmr_terms[term_id] = term

        ### Create term embeddings
        term_embeddings = {'id':[], 'embs':[]}
        for term_id, term_data in tqdm(pmr_terms.items()):
            term_embeddings['id'] += [term_id]
            term_embeddings['embs'] += [to_embedding(term_data, self.__biobert_model, self.__nlp_model)]
        
        # prepare for cellml_embeddings
        cellml_embeddings = {'id':[], 'embs':[]}
        tmp_term_embeddings = dict(zip(term_embeddings['id'], term_embeddings['embs']))

        def get_term_embedding(term_id, term_data):
            if term_id in tmp_term_embeddings:
                return tmp_term_embeddings[term_id]
            tmp_term_embeddings[term_id] = to_embedding(term_data, self.__biobert_model, self.__nlp_model)
            return tmp_term_embeddings[term_id]


        # prepare for doc_term_embeddings
        umls_types = ({'T0'+str(i) for i in range(16,32)} | {'T0'+str(i) for i in range(38,46)}) - {'T041', 'T028'}
        doc_termids = {}
        doc_ents = {}
        all_documentation = list(set(all_documentation))
        docs = self.__nlp_model.pipe(all_documentation)
        all_ents, abbrs = [], {}
        not_confirm_terms = {}
        for idx, doc in enumerate(docs):
            doc_termids[all_documentation[idx]] = {'confirm':[], 'not_confirm':[]}
            doc_ents[all_documentation[idx]] = [ent.text for ent in doc.ents]
            all_ents += doc_ents[all_documentation[idx]]
            # update abbreviation
            for abrv in doc._.abbreviations:
                abbrs[abrv.text] = abrv._.long_form.text

            for ent in doc.ents:
                # converting abbreviation to long-term
                ent_txt = abbrs[ent.text] if ent.text in abbrs else ent.text
                # cannot justify the correct classification if the entity text is too short:
                if len(ent_txt) <= 2 and '// ' in ent_txt:
                    continue
            
                # term_id = kb_ent[0]
                # link = linker.kb.cui_to_entity[term_id]
                
                # extract from ontology lookup
                cand = self.__get_sckan_candidate(ent_txt)

                if cand is not None:
                    term_id = cand[0]
                    doc_termids[all_documentation[idx]]['confirm'] += [term_id]
                else:
                    for kb_ent in ent._.kb_ents:
                    # extract from umls
                        term_id = kb_ent[0]
                        link = self.__linker.kb.cui_to_entity[term_id]
                        if len(set(link.types) & umls_types) > 0:
                            # add terms
                            label = link.canonical_name
                            definition = link.definition if link.definition is not None else []
                            synonym = link.aliases if link.aliases is not None else []
                            is_a = link.types
                            is_a_text = [self.__linker.kb.semantic_type_tree.get_canonical_name(t) for t in is_a]
                            term_data = {'label':label, 'def':definition, 'synonym':synonym, 'is_a':is_a, 'is_a_text':is_a_text}
                            get_term_embedding(term_id, term_data)
                            doc_termids[all_documentation[idx]]['not_confirm'] += [term_id]
                            not_confirm_terms[term_id] = term_data
        
        # now checking all cellml
        all_ents = list(set(all_ents))
        doc_embeddings = self.__biobert_model.encode(all_documentation, convert_to_tensor=True)
        ent_embeddings = self.__biobert_model.encode(all_ents, convert_to_tensor=True)
        
        # exploring cellml files for any documentation
        for cellml_url in tqdm(new_cellmls):
            cellml_id = cellmls[cellml_url]
            if len(cellml_ents:=[ent for doc in cellml_id.get('documentation',[]) for ent in doc_ents[doc]]) == 0:
                if len(cellml_id.get('documentation',[])) > 0:
                    # condition where there is documentation but no entity detected
                    cellml_emb = torch.mean(torch.stack([doc_embeddings[all_documentation.index(doc)] for doc in cellml_id['documentation']]), dim=0)
                elif len(rdf_leaves:=cellml_id.get('rdfLeaves', [])) > 0:
                    # condition where there is no documentation
                    embs = [term_embeddings['embs'][term_embeddings['id'].index(url_to_curie(o))] for o in rdf_leaves if url_to_curie(o) in term_embeddings['id']]
                    if len(embs) > 0:
                        cellml_emb = torch.mean(torch.stack(embs), dim=0)
                    else: # no relevant information to be use, ignore
                        continue
                else:
                    # no rdfleaves and documentation, ignore it
                    continue
            else:
                # condition where entity from documentation is available
                cellml_emb = torch.mean(torch.stack([ent_embeddings[all_ents.index(cellml_ent)] for cellml_ent in cellml_ents]), dim=0)
            # check not_confirm term_id
            for doc in cellml_id.get('documentation',[]):
                for term_id in set(doc_termids[doc]['not_confirm']):
                    if util.cos_sim(tmp_term_embeddings[term_id], cellml_emb) >= 0.9:
                        pmr_terms[term_id] = not_confirm_terms[term_id]
                        term_embeddings['id'] += [term_id]
                        term_embeddings['embs'] += [tmp_term_embeddings[term_id]]
                        if cellml_url not in cellml_to_terms:
                            cellml_to_terms[cellml_url] = []
                        cellml_to_terms[cellml_url] += [term_id]
                for term_id in set(doc_termids[doc]['confirm']):
                    pmr_terms[term_id] = self.__sckan_terms[term_id]
                    term_embeddings['id'] += [term_id]
                    idx = self.__sckan_biobert_embs['id'].index(term_id)
                    term_emb = self.__sckan_biobert_embs['embs'][idx]
                    term_embeddings['embs'] += [term_emb]
                    if cellml_url not in cellml_to_terms:
                        cellml_to_terms[cellml_url] = []
                    cellml_to_terms[cellml_url] += [term_id]

            cellml_embeddings['id'] += [cellml_url]
            cellml_embeddings['embs'] += [cellml_emb]

        # modify cellml_to_term as term_to_cellml
        term_to_cellml = {}
        for cellml_url, cellml_terms in cellml_to_terms.items():
            for term_id in cellml_terms:
                if term_id not in term_to_cellml:
                    term_to_cellml[term_id] = []
                term_to_cellml[term_id] += [cellml_url]

        # dump workspaces
        dumpJson(workspaces, RS_WORKSPACE)

        # consolidate with the available search_data if not clean extraction
        if not clean_extraction and search_data.get('term') is not None:
            for i, term_id in enumerate(term_embeddings['id']):
                if term_id not in search_data['term']:
                    search_data['term'] += [term_id]
                    search_data['embedding'] = torch.cat((search_data['embedding'], torch.stack([term_embeddings['embs'][i]])), 0)
            for term_id, val in pmr_terms.items():
                if term_id not in search_data['pmrTerm']:
                    search_data['pmrTerm'][term_id] = val
            for term_id, cellml_id in term_to_cellml.items():
                if term_id not in search_data['termCellml']:
                    search_data['termCellml'][term_id] = [cellml_id]
                elif cellml_id not in search_data['termCellml'][term_id]:
                    search_data['termCellml'][term_id] += [cellml_id]
            for i, cellml_id in enumerate(cellml_embeddings['id']):
                if cellml_id not in search_data['cellmlId']:
                    search_data['cellmlId'] += [cellml_id]
                    search_data['cellmlEmbs'] = torch.cat((search_data['cellmlEmbs'], torch.stack([cellml_embeddings['embs'][i]])), 0)

            return {'id':search_data['term'], 'embs':search_data['embedding']}, search_data['pmrTerm'], cellmls, search_data['termCellml'], {'id':search_data['cellmlId'], 'embs':search_data['cellmlEmbs']}
        # return results
        term_embeddings['embs'] = torch.stack(term_embeddings['embs'])
        cellml_embeddings['embs'] = torch.stack(cellml_embeddings['embs'])
        return term_embeddings, pmr_terms, cellmls, term_to_cellml, cellml_embeddings

    def create_search_index(self, clean_extraction=True):
        self.__sckan_terms, self.__sckan_bert_embs, self.__sckan_biobert_embs = extract_sckan_terms(
            ontologies=self.__ontologies,
            to_embedding=to_embedding, 
            bert_model=self.__bert_model, 
            biobert_model=self.__biobert_model,
            nlp_model=self.__nlp_model,
            sckan_version=self.__sckan_version,
            clean_extraction=clean_extraction,
            )
        
        log.info(f'SCKAN has {self.__sckan_bert_embs["embs"].shape} terms of {BERTModel} and {self.__sckan_biobert_embs["embs"].shape} terms of {BIOBERT}')

        term_embeddings, pmr_terms, cellmls, term_to_cellml, cellml_embeddings = self.__extract_pmr(clean_extraction)

        pmr_clusterer = CellmlClusterer(workspace_dir=self.__pmr_workspace_dir, cellmls=cellmls)

        combined_data = {
            'term':term_embeddings['id'],                       # a list of term having embedding
            'embedding':term_embeddings['embs'],   # an array of term embedding
            'pmrTerm':pmr_terms,                                # a dictionary of terms in the PMR
            'sckanTerm':self.__sckan_terms,                     # a dictionary of terms in SCKAN
            'cellml':cellmls,                                   # a dictionary of CellML files
            'cluster':pmr_clusterer.getDict(),                  # a dictionary of CellML cluster
            'termCellml':term_to_cellml,                        # a dictionay terms in CellML files
            'cellmlId': cellml_embeddings['id'],                # a list of CellML files having embedding
            'cellmlEmbs':cellml_embeddings['embs'] # an array of CellML embedding
            }

        ### safe to file
        torch.save(combined_data, SEARCH_FILE)

        ### updating METADATA
        from .. import __version__
        METADATA['pmrindexer_version'] = __version__
        dumpJson(METADATA, METADATA_FILE)
