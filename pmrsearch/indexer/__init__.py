import json
import os
import gzip
import pickle
import pandas as pd
import re
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import codecs
import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from rdflib.namespace import OWL, RDFS
import logging
import rdflib

from ..setup import RESOURCE_PATH, SCKAN_FILE, SEARCH_FILE, BERTModel, BIOBERT, NLPModel, METADATA, METADATA_FILE, SCKAN_BERT_FILE, SCKAN_TERMS, url_to_curie
from .sckan_crawler import extract_sckan_terms

ONTO_DF = f'{RESOURCE_PATH}/ontoDf.gz'
SCKAN_PICKLE = f'{RESOURCE_PATH}/sckan.graph'

RS_VARIABLE = f'{RESOURCE_PATH}/listOfVariable.json'
RS_CELLML = f'{RESOURCE_PATH}/listOfCellml.json'
RS_WORKSPACE = f'{RESOURCE_PATH}/listOfWorkspace.json'
RS_CLUSTER = f'{RESOURCE_PATH}/cellmlClusterer.json'
RS_CELLML_DATA = f'{RESOURCE_PATH}/cellml_data.json'

ALPHA = 0.5
BETA = 0.5

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
    def __init__(self, pmr_workspace_dir, bert_model=None, biobert_model=None, nlp_model=None, sckan_url=None, pmr_onto=ONTO_DF):
        self.__bert_model = SentenceTransformer(BERTModel) if bert_model is None else bert_model 
        self.__biobert_model = SentenceTransformer(BIOBERT) if biobert_model is None else biobert_model
        self.__nlp_model = spacy.load(NLPModel) if nlp_model is None else nlp_model
        self.__nlp_model.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": 0.9,})
        self.__linker = self.__nlp_model.get_pipe("scispacy_linker")
        self.__nlp_model.add_pipe("abbreviation_detector")
        self.__pmr_workspace_dir = pmr_workspace_dir
        self.__sckan_url = sckan_url
        with gzip.GzipFile(pmr_onto, 'rb') as f:
            self.__ontologies = pickle.load(f)

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
            r1 = self.__sckan_search(query, self.__bert_model, self.__sckan_bert_embs, k=10, th=0.6)
            r2 = self.__sckan_search(query, self.__biobert_model, self.__sckan_biobert_embs, k=10, th=0.9)
            for r2_ids in r2:
                if r2_ids in r1:
                    return (r2_ids, r2[r2_ids][0], (r2[r2_ids][1]+r1[r2_ids][1])/2)
                
    def __extract_pmr(self):
        ### Load variables PMR and organised in term_to_vars and var_to_terms dictionaries
        
        terms, term_to_vars, var_to_terms = {}, {}, {}

        with open(RS_VARIABLE, 'r') as fp:
            variables = json.load(fp)

        for var_id, value in tqdm(variables['data'].items()):
            if 'rdfLeaves' in value:
                var_to_terms[var_id] = []
                for leaf in value['rdfLeaves']:
                    term_id = url_to_curie(str(leaf))
                    if term_id is not None and term_id in self.__ontologies.index:
                        term_to_vars.setdefault(term_id, [])
                        term_to_vars[term_id] += [var_id]
                        var_to_terms[var_id] += [term_id]
                if len(var_to_terms[var_id]) == 0:
                    del var_to_terms[var_id]

        for term_id in term_to_vars:
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
            terms[term_id] = term

        ### Create term embeddings
        term_embeddings = {'id':[], 'embs':[]}
        for term_id, term_data in tqdm(terms.items()):
            term_embeddings['id'] += [term_id]
            term_embeddings['embs'] += [to_embedding(term_data, self.__biobert_model, self.__nlp_model)]

        ### Load cellmls and organised into cellml_to_terms and term_to_cellml
        cellml_to_terms, term_to_cellml, id_to_cellml = {}, {}, {}
        
        for cellml_id, cellml in tqdm(self.__cellmls['data'].items()):
            id_to_cellml[cellml['id']] = cellml_id
            cellml_to_terms[cellml_id] = [] 
            if 'variables' not in cellml:
                continue
            for var_id in cellml['variables']:
                if var_id in var_to_terms:
                    cellml_to_terms[cellml_id] = cellml_to_terms.get(cellml_id, []) + var_to_terms[var_id]
                    for term_id in var_to_terms[var_id]:
                        if term_id not in term_to_cellml:
                            term_to_cellml[term_id] = []
                        term_to_cellml[term_id] += [cellml_id]
            cellml_to_terms[cellml_id] = list(set(cellml_to_terms[cellml_id]))

        # prepare for cellml_embeddings
        cellml_embeddings = {'id':[], 'embs':[]}
        tmp_term_embeddings = dict(zip(term_embeddings['id'], term_embeddings['embs']))

        def get_term_embedding(term_id, term_data):
            if term_id in tmp_term_embeddings:
                return tmp_term_embeddings[term_id]
            tmp_term_embeddings[term_id] = to_embedding(term_data, self.__biobert_model, self.__nlp_model)
            return tmp_term_embeddings[term_id]
        
        # exploring cellml files
        
        umls_types = ({'T0'+str(i) for i in range(16,32)} | {'T0'+str(i) for i in range(38,46)}) - {'T041', 'T028'}
        # pattern to handle abbreviations
        pattern = r'\((\w+)s\)'

        for cellml_id, cellml in tqdm(self.__cellmls['data'].items()):
            cellml_file = os.path.join(self.__pmr_workspace_dir, cellml['workingDir'], cellml['cellml'])
            cellml_dir = os.path.dirname(cellml_file)
            documentation = []
            for file_name in os.listdir(cellml_dir):
                if file_name.endswith('.html') or file_name.endswith('.md') or file_name.endswith('.rst') or cellml['cellml'] == file_name:
                    html_path = os.path.join(cellml_dir,file_name)
                    try:
                        file = codecs.open(html_path, "r", "utf-8")
                        soup = BeautifulSoup(file.read(), 'lxml')
                    except Exception:
                        file = codecs.open(html_path, "r", "latin-1")
                        soup = BeautifulSoup(file.read(), "lxml")
                    
                    if soup.title is not None:
                        title = re.sub(pattern, r'(\1)', str(soup.title.text))
                        if title != '':
                            documentation += [title]
                    
                    paragraphs = soup.find_all('p') + soup.find_all('para')
                    for p in paragraphs:
                        if p is None:
                            try:
                                p = re.sub('\s+', ' ', p.text).strip()
                                p = re.sub(pattern, r'(\1)', p)
                                if len(p.split()) > 50 and len(p) > 250:
                                    documentation += p.split('. ')
                            except Exception:
                                logging.warning(f'Cannot load a paragraph from {file_name}')

            documentation = list(set(documentation))
                    
            # process if documentation is available
            if len(documentation)>0:
                cellml['documentation'] = documentation
                docs = self.__nlp_model.pipe(documentation)
                ents, abbrs = {}, {}
                for doc in docs:
                    for ent in doc.ents:
                        ents[ent.text] = ent
                    for abrv in doc._.abbreviations:
                        abbrs[abrv.text] = abrv._.long_form.text
                
                # create cellml embedding as a combination of entity embeddings
                if len(ents) == 0:
                    cellml_emb = torch.mean(self.__biobert_model.encode(documentation, convert_to_tensor=True), 0)
                else:
                    cellml_emb = torch.mean(self.__biobert_model.encode(list(ents.keys()), convert_to_tensor=True), 0)
                
                for ent_txt, ent in ents.items():
                    # converting abbreviation to long-term
                    ent_txt = abbrs[ent_txt] if ent_txt in abbrs else ent_txt
                    
                    # cannot justify the correct classification if the entity text is too short:
                    if len(ent_txt) <= 2 and '// ' in ent_txt:
                        continue
        #             term_id = kb_ent[0]
        #             link = linker.kb.cui_to_entity[term_id]
                    
                    # extract from ontology lookup
                    cand = self.__get_sckan_candidate(ent_txt)

                    if cand is not None:
                        term_id = cand[0]
                        # get the embedding first
                        terms[term_id] = self.__sckan_terms[term_id]
                        term_embeddings['id'] += [term_id]
                        idx = self.__sckan_biobert_embs['id'].index(term_id)
                        term_emb = self.__sckan_biobert_embs['embs'][idx]
                        term_embeddings['embs'] += [term_emb]
                        cellml_to_terms[cellml_id] += [term_id]
                        if term_id not in term_to_cellml:
                            term_to_cellml[term_id] = []
                        term_to_cellml[term_id] += [cellml_id]

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
                                term_emb = get_term_embedding(term_id, term_data)
                                if util.cos_sim(term_emb, cellml_emb) >= 0.9:
                                    terms[term_id] = term_data
                                    term_embeddings['id'] += [term_id]
                                    term_embeddings['embs'] += [term_emb]
                                    cellml_to_terms[cellml_id] += [term_id]
                                    if term_id not in term_to_cellml:
                                        term_to_cellml[term_id] = []
                                    term_to_cellml[term_id] += [cellml_id]

                cellml_embeddings['id'] += [cellml_id]
                cellml_embeddings['embs'] += [cellml_emb]
        
        return terms, term_to_cellml, id_to_cellml, term_embeddings, cellml_embeddings

    def create_search_index(self):
        self.__sckan_terms, self.__sckan_bert_embs, self.__sckan_biobert_embs = extract_sckan_terms(
            ontologies=self.__ontologies,
            to_embedding=to_embedding, 
            bert_model=self.__bert_model, 
            biobert_model=self.__biobert_model,
            nlp_model=self.__nlp_model,
            url=self.__sckan_url, 
            )
        
        with open(RS_CELLML, 'r') as fp:
            self.__cellmls = json.load(fp)

        pmr_terms, term_to_cellml, id_to_cellml , term_embeddings, cellml_embeddings = self.__extract_pmr()

        ### enrich cellml with exposure and commit sha
        with open(RS_WORKSPACE, 'r') as fp:
            self.__workspaces = json.load(fp)

        # providing exposure and commit sha
        for _, workspace in self.__workspaces['data'].items():
            for short_id in workspace.get('cellml', []):
                if short_id in id_to_cellml:
                    cellml_id = id_to_cellml[short_id]
                    self.__cellmls['data'][cellml_id]['sha'] = workspace['commit']
                    if 'exposures' in workspace:
                        exp_id = list(workspace['exposures'].keys())[-1]
                        self.__cellmls['data'][cellml_id]['exposure'] = exp_id

        term_embeddings['embs'] = torch.stack(term_embeddings['embs'])
        cellml_embeddings['embs'] = torch.stack(cellml_embeddings['embs'])

        with open(RS_CLUSTER, 'r') as f:
            cluster = json.load(f)

        combined_data = {
            'term':term_embeddings['id'], 
            'embedding':term_embeddings['embs'], 'pmrTerm':pmr_terms, 
            'sckanTerm':self.__sckan_terms, 
            'cellml':self.__cellmls, 
            'cluster':cluster, 
            'termCellml':term_to_cellml, 
            'cellmlId': cellml_embeddings['id'], 
            'cellmlEmbs':cellml_embeddings['embs']
            }

        ### safe to file
        torch.save(combined_data, SEARCH_FILE)

        ### updating METADATA
        from .. import __version__
        METADATA['pmrindexer_version'] = __version__
        with open(METADATA_FILE, 'w') as f:
            json.dump(METADATA, f)
