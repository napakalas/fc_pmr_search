#===============================================================================

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import spacy
import json
import os
import sqlite3
import logging

#===============================================================================

from ..setup import SEARCH_FILE, SCKAN_FILE, PMR_URL, BIOBERT, NLPModel, SCKAN2PMR, METADATA, METADATA_FILE, SCKAN2PMR_SQLITE

TOP_K = 1000
MIN_SIM = 0.6
C_WEIGHT = 0.8

#===============================================================================

class PMRSearcher:
    def __init__(self, emb_model=None, nlp_model=None):
        data = torch.load(SEARCH_FILE)
        self.__term_embs = data['embedding']
        self.__terms = data['term']
        self.__pmr_term = data['pmrTerm']
        self.__sckan_term = data['sckanTerm']
        self.__cellmls = data['cellml']        
        self.__cluster = data['cluster']
        self.__term_cellmls = data['termCellml']
        self.__cellml_ids = data['cellmlId']
        self.__cellml_embs = data['cellmlEmbs']

        self.__model = SentenceTransformer(BIOBERT) if emb_model is None else emb_model
        self.__nlp = spacy.load(NLPModel) if nlp_model is None else nlp_model

        
        data = torch.load(SCKAN_FILE)
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
            cellml = self.__cellmls['data'][cellml_id]
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
            cellml = self.__cellmls['data'][cellml_id]
            if 'workspace' in cellml:
                workspaces += [cellml['workspace']] if cellml['workspace'] not in workspaces else []
            if 'exposure' in self.__cellmls['data'][cellml_id]:
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
            if 'exposure' in self.__cellmls['data'][cellml_id]:
                return self.__cellmls['data'][cellml_id]['exposure']
            else:
                cluster_id = self.__cluster['url2Cluster'][cellml_id]
                if cluster_id != '-1':
                    similar_cellmls = self.__cluster['cluster'][cluster_id]
                    for sim_cellml_id in similar_cellmls:
                        if 'exposure' in self.__cellmls['data'][sim_cellml_id]:
                            return self.__cellmls['data'][sim_cellml_id]['exposure']
            return ''

        top_results = self.__search_terms(query, self.__cellml_embs, context, c_weight)
        cellml_res, selected_cellmls = [], []
        for _, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            if score < min_sim or len(cellml_res) >= topk:
                break
            cellml_id = self.__cellml_ids[idx]
            if cellml_id not in selected_cellmls:
                selected_cellmls += [cellml_id]
                cellml = self.__cellmls['data'][cellml_id]
                cellml_res += [{'score':score.item(), 
                                'cellml':[cellml_id], 
                                'workspace':[cellml['workspace']], 
                                'exposure':[get_exposure(cellml_id)]}]
        return cellml_res
    
    def generate_term_to_pmr(self, min_sim=MIN_SIM):
        term2pmr = {}
        logging.info('Generating mapp of SCKAN terms to PMR models')
        for sckan_id in tqdm(self.__sckan_term.keys()):
            if len(pmr_models := self.search_cellml(sckan_id, min_sim=min_sim)) > 0:
                found_models = []
                for pmr_model in pmr_models:
                    cellml = self.__cellmls['data'][pmr_model[1]]
                    sha = cellml['sha']
                    cellml_link = PMR_URL + pmr_model[1].replace('HEAD', sha)
                    new_model = {'cellml': cellml_link}
                    new_model['workspace'] = PMR_URL + cellml['workspace']
                    if 'exposure' in cellml:
                        new_model['exposure'] = PMR_URL + cellml['exposure']
                    new_model['score'] = pmr_model[0]
                    found_models += [new_model]
                if len(found_models) > 0:
                    term2pmr[sckan_id] = found_models
        return term2pmr


    def generate_term_to_pmr_save(self, min_sim=MIN_SIM):
        term2pmr = self.generate_term_to_pmr(min_sim=min_sim)

        # save to json file
        with open(SCKAN2PMR, 'w') as f:
            json.dump(term2pmr, f, indent=4)

        # with open(SCKAN2PMR, 'r') as f:
        #     term2pmr = json.load(f)

        # update METADATA
        METADATA['minimum_similarity'] = min_sim
        with open(METADATA_FILE, 'w') as f:
            json.dump(METADATA, f)

        # save to sqllite
        if os.path.exists(SCKAN2PMR_SQLITE):
            os.remove(SCKAN2PMR_SQLITE)
        conn = sqlite3.connect(SCKAN2PMR_SQLITE)
        cursor = conn.cursor()
        SQL_SCKAN2PMR = """
            CREATE TABLE SCKAN2PMR
            (
                SCKAN_ID    TEXT    NOT NULL,
                CELLML      TEXT    NOT NULL,
                WORKSPACE   TEXT    NOT NULL,
                EXPOSURE    TEXT,
                SCORE       REAL
            );
        """
        cursor.execute(SQL_SCKAN2PMR)
        SQL_METADATA = """
            CREATE TABLE METADATA
            (
                SCKAN_URL           TEXT    NOT NULL,
                SCKAN_BUILD         TEXT    NOT NULL,
                PMRINDEXER_VERSION  TEXT    NOT NULL,
                MINIMUM_SIMILARITY  TEXT    NOT NULL
            );
        """
        cursor.execute(SQL_METADATA)
        for sckan_id, pmr_models in tqdm(term2pmr.items()):
            data = []
            for pmr in pmr_models:
                data += [(sckan_id, pmr['cellml'], pmr['workspace'], pmr.get('exposure'), pmr['score'])]
            cursor.executemany(
                "INSERT INTO SCKAN2PMR values (?, ?, ?, ?, ?)", data
            )
            conn.commit()
        cursor.execute(
            "INSERT INTO METADATA values (?, ?, ?, ?)",
            (METADATA['sckan_url'], METADATA['sckan_build'], METADATA['pmrindexer_version'], METADATA['minimum_similarity'])
        )
        conn.commit()
        conn.close()

#===============================================================================
