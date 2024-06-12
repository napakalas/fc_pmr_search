#===============================================================================

import torch
from sentence_transformers import SentenceTransformer, util
import torch.backends
import torch.backends.mps
from tqdm import tqdm
import spacy
import json
import os
import logging as log
from jsonschema import validate
from jsonschema.exceptions import ValidationError

#===============================================================================

from ..setup import SEARCH_FILE, SCKAN_BIOBERT_FILE, CURRENT_PATH, BIOBERT, NLPModel, SCKAN2PMR, METADATA, METADATA_FILE, SCHEMA_METADATA_FILE, SCHEMA_SCKAN2PMR_FILE
from ..setup import dumpJson, loadJson

TOP_K = 1000
MIN_SIM = 0.6
C_WEIGHT = 0.8

#===============================================================================

class PMRSearcher:
    def __init__(self, emb_model=None, nlp_model=None):
        device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        map_location = torch.device(device)
        data = torch.load(SEARCH_FILE, map_location=map_location)
        self.__term_embs = data['embedding']
        self.__terms = data['term']
        self.__pmr_term = data['pmrTerm']
        self.__sckan_term = data['sckanTerm']
        self.__cellmls = data['cellml']        
        self.__cluster = data['cluster']
        self.__term_cellmls = data['termCellml']
        self.__cellml_ids = data['cellmlId']
        self.__cellml_embs = data['cellmlEmbs']

        self.__model = SentenceTransformer(BIOBERT, device=device) if emb_model is None else emb_model
        self.__nlp = spacy.load(NLPModel) if nlp_model is None else nlp_model
        
        data = torch.load(SCKAN_BIOBERT_FILE, map_location=map_location)
        self.__sckan_ids = data['id']
        self.__sckan_embs = data['embs']
        
    def __get_query_embedding(self, query, context=None, c_weight=C_WEIGHT):
        if query in self.__sckan_ids:
            query_emb = self.__sckan_embs[self.__sckan_ids.index(query)]
        else:
            # query is separate as different entities and combine them using averaging
            doc = self.__nlp(query)
            if len(doc.ents) > 0:
                ents = [ent.text for ent in doc.ents]# + [query]
            else:
                ents = [query]
            query_emb = torch.mean(self.__model.encode(ents, convert_to_tensor=True), 0)
        context = context if isinstance(context, list) else [context] if isinstance(context, str) else None
        if context is not None and c_weight > 0:
            context_emb1 = torch.mean(self.__model.encode(context, convert_to_tensor=True), 0) * c_weight
            if query in self.__sckan_term:
                txt = ' '.join([self.__sckan_term[query]['label']] + context)
            else:
                txt = ' '.join([query] + context)
            context_emb2 = self.__model.encode(txt, convert_to_tensor=True) * c_weight
            query_emb = (query_emb + context_emb1 + context_emb2) / (1 + c_weight*2)
        return query_emb
    
    def __search_terms(self, query, embs, context=None, c_weight=C_WEIGHT):
        query_emb = self.__get_query_embedding(query, context, c_weight)
        cos_scores = util.pytorch_cos_sim(query_emb, embs)[0]
        top_results = torch.topk(cos_scores, 1000)
        return top_results
    
    def __get_wks_exp(self, term):
        cellmls = self.__term_cellmls[term]
        workspaces, exposures = [], []
        for cellml_id in cellmls:
            cellml = self.__cellmls[cellml_id]
            if cellml['workspace'] not in workspaces:
                workspaces += [cellml['workspace']]
            if 'exposure' in cellml:
                if cellml['exposure'] not in exposures:
                    exposures += [cellml['exposure']]

        if len(exposures) > 0:
            return {'exposure':exposures, 'workspace':workspaces, 'cellml':cellmls}
        
        # check other model if exposure is not available
        available_cellml = []
        for cellml_id in self.__term_cellmls[term]:
            cluster_id = self.__cluster['url2Cluster'][cellml_id]
            if cluster_id == '-1':
                if cellml_id not in available_cellml:
                    available_cellml += [cellml_id]
            else:
                similar_cellmls = self.__cluster['cluster'][cluster_id]
                available_cellml += [idx for idx in similar_cellmls if idx not in available_cellml]

        # get exposure and workspace
        exposures, workspaces = [], []
        for cellml_id in available_cellml:
            cellml = self.__cellmls[cellml_id]
            if 'workspace' in cellml:
                workspaces += [cellml['workspace']] if cellml['workspace'] not in workspaces else []
            if 'exposure' in self.__cellmls[cellml_id]:
                exposures += [cellml['exposure']] if cellml['exposure'] not in exposures else []
        return {'exposure':exposures, 'workspace':workspaces, 'cellml':available_cellml}
    
    def search(self, query, context=None, topk=TOP_K, min_sim=MIN_SIM, c_weight=0.5):
        top_results = self.__search_terms(query, self.__term_embs, context, c_weight)
        
        #get topk results
        cellml_res, selected_terms = [], []
        for _, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            if score < min_sim or len(cellml_res) >= topk:
                break
            term = self.__terms[idx]
            if term.startswith('C'):
                continue
            if term not in selected_terms and term in self.__term_cellmls:
                selected_terms += [term]
                rst = {'score': (score.item(), term, self.__pmr_term[term]['label'])}
                rst.update(self.__get_wks_exp(term))
                cellml_res += [rst]
        return cellml_res
    
    def __search_object(self, query, context=None, topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT, obj_type='cellmls'):
        top_results = self.__search_terms(query, self.__term_embs, context, c_weight)
        object_res, selected_objects = [], []
        for _, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            if score < min_sim or len(object_res) >= topk:
                break
            term = self.__terms[idx]
            if  term in self.__term_cellmls:
                objects = self.__get_wks_exp(term)[obj_type]
                for obj in objects:
                    if obj not in selected_objects:
                        selected_objects += [obj]
                        object_res += [(score.item(), obj)]
        return object_res
    
    def search_exposure(self, query, context=None, topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        return self.__search_object(query, context, topk, min_sim, c_weight, obj_type='exposure')
    
    def search_workspace(self, query, context=None, topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        return self.__search_object(query, context, topk, min_sim, c_weight, obj_type='workspace')

    def search_cellml(self, query, context=None, topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        return self.__search_object(query, context, topk, min_sim, c_weight, obj_type='cellml')

    def search_all(self, query, context=None, topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        return {
            'exposure': self.__search_object(query, context, topk, min_sim, c_weight, obj_type='exposure'),
            'workspace': self.__search_object(query, context, topk, min_sim, c_weight, obj_type='workspace'),
            'cellml': self.__search_object(query, context, topk, min_sim, c_weight, obj_type='cellml')
        }
    
    def search_by_cellml(self, query, context=None, topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        def get_exposure(cellml_id):
            if 'exposure' in self.__cellmls[cellml_id]:
                return self.__cellmls[cellml_id]['exposure']
            else:
                cluster_id = self.__cluster['url2Cluster'][cellml_id]
                if cluster_id != '-1':
                    similar_cellmls = self.__cluster['cluster'][cluster_id]
                    for sim_cellml_id in similar_cellmls:
                        if 'exposure' in self.__cellmls[sim_cellml_id]:
                            return self.__cellmls[sim_cellml_id]['exposure']
            return ''

        top_results = self.__search_terms(query, self.__cellml_embs, context, c_weight)
        cellml_res, selected_cellmls = [], []
        for _, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            if score < min_sim or len(cellml_res) >= topk:
                break
            cellml_id = self.__cellml_ids[idx]
            if cellml_id not in selected_cellmls:
                selected_cellmls += [cellml_id]
                cellml = self.__cellmls[cellml_id]
                cellml_res += [{'score':score.item(), 
                                'cellml':[cellml_id], 
                                'workspace':[cellml['workspace']], 
                                'exposure':[get_exposure(cellml_id)]}]
        return cellml_res
    
    def check_and_generate_annotation_completeness(self, annotation_file):
        missed = {}
        found = {}
        included = ['FTU Name', 'Organ', 'Model', 'Label', 'Nerve Name', 'Organ/System', 'Models', 'Organ Name', 'Systems', 'System Name', 'Vessel Name']
        if annotation_file is None:
            pass
        elif os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            for level, values in annotations.items():
                for value in values:
                    new_value = {k:v for k, v in value.items() if k in included}
                    term_id = value.get('Model', value.get('Models', ''))
                    if term_id not in self.__sckan_term:
                        missed[term_id] += [new_value]
                    else:
                        found[level] += [new_value]
        from .. import __version__
        anatomical_terms = f'{CURRENT_PATH}/output/anatomical_terms-{__version__}.json'
        with open(anatomical_terms, 'w') as f:
            json.dump({'found':found, 'missed':missed}, f, indent=4)

    def generate_term_to_pmr(self, min_sim=MIN_SIM, exposure_only=True):
        term2pmr = []
        log.info('Generating map of SCKAN terms to PMR models')
        for sckan_id in tqdm(self.__sckan_term.keys()):
            if len(pmr_models := self.search_cellml(sckan_id, min_sim=min_sim)) > 0:
                found_models = []
                for pmr_model in pmr_models:
                    if pmr_model[1] not in found_models:
                        cellml = self.__cellmls[pmr_model[1]]
                        new_model = {
                            'cellml': pmr_model[1],
                            'workspace': cellml['workspace'],
                            'score': pmr_model[0]
                            }
                        if exposure_only and 'exposure' in cellml:
                            new_model['exposure'] = cellml['exposure']
                            found_models += [new_model]
                        elif not exposure_only:
                            if 'exposure' in cellml:
                                new_model['exposure'] = cellml['exposure']
                            found_models += [new_model]
                if len(found_models) > 0:
                    term2pmr += [
                        {
                            "sckan_term": sckan_id,
                            "cellmls": found_models
                        }
                    ]
        return term2pmr

    def generate_term_to_pmr_save(self, min_sim=MIN_SIM, exposure_only=True):
        term2pmr = self.generate_term_to_pmr(min_sim=min_sim, exposure_only=exposure_only)
        # save to json file
        dumpJson(term2pmr, SCKAN2PMR)
        self.__validate_json(term2pmr, SCHEMA_SCKAN2PMR_FILE)

        # update METADATA
        METADATA['minimum_similarity'] = min_sim
        from .. import __version__
        METADATA['pmrindexer_version'] = __version__
        dumpJson(METADATA, METADATA_FILE)
        self.__validate_json(METADATA, SCHEMA_METADATA_FILE)

    def __validate_json(self, data, schema_file):
        schema = loadJson(schema_file)
        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            print(e)

#===============================================================================
