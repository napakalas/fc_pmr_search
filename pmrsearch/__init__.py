#===============================================================================

import torch
from sentence_transformers import SentenceTransformer, util

#===============================================================================

SEARCH_DATA = 'indexes/search_data.pt'
SCKAN_DATA = 'indexes/sckan.pt'
BERTModel = 'gsarti/biobert-nli'
TOP_K = 5
MIN_SIM = 0.5
C_WEIGHT = 0.8
MAX_RETRIEVE = 1000

#===============================================================================

class PMRSearcher:
    def __init__(self):
        data = torch.load(SEARCH_DATA)
        self.__term_embs = data['embedding']
        self.__terms = data['term']
        self.__pmr_term = data['pmrTerm']
        self.__sckan_term = data['sckanTerm']
        self.__cellmls = data['cellml']        
        self.__cluster = data['cluster']
        self.__term_cellmls = data['termCellml']
        self.__cellml_ids = data['cellmlId']
        self.__cellml_embs = data['cellmlEmbs']
        self.__model = SentenceTransformer(BERTModel)
        
        data = torch.load(SCKAN_DATA)
        self.__sckan_ids = data['id']
        self.__sckan_embs = data['embs']
        
    def __get_query_embedding(self, query, context=[], c_weight=C_WEIGHT):
        if query in self.__sckan_ids:
            query_emb = self.__sckan_embs[self.__sckan_ids.index(query)]
        else:
            query_emb = self.__model.encode(query, convert_to_tensor=True)  
        context = context if isinstance(context, list) else [context]
        if len(context) > 0 and c_weight > 0:
            context_emb1 = torch.mean(self.__model.encode(context, convert_to_tensor=True), 0) * c_weight
            if query in self.__sckan_term:
                txt = ' '.join([self.__sckan_term[query]['label']] + context)
            else:
                txt = ' '.join([query] + context)
            context_emb2 = self.__model.encode(txt, convert_to_tensor=True) * c_weight
            query_emb = (query_emb + context_emb1 + context_emb2) / (1 + c_weight*2)
        return query_emb
    
    def __search_terms(self, query, embs, context=[], c_weight=C_WEIGHT):
        query_emb = self.__get_query_embedding(query, context, c_weight)
        cos_scores = util.pytorch_cos_sim(query_emb, embs)[0]
        top_results = torch.topk(cos_scores, MAX_RETRIEVE)
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
    
    def search(self, query, context=[], topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        top_results = self.__search_terms(query, self.__term_embs, context, c_weight)
        
        #get topk results
        cellml_res, selected_terms = [], []
        for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
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
    
    def __search_object(self, query, context=[], topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT, obj_type='exposure'):
        top_results = self.__search_terms(query, self.__term_embs, context, c_weight)
        object_res, selected_objects = [], []
        for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
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
    
    def search_exposure(self, query, context=[], topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        return self.__search_object(query, context, topk, min_sim, c_weight, obj_type='exposure')
    
    def search_workspace(self, query, context=[], topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        return self.__search_object(query, context, topk, min_sim, c_weight, obj_type='workspace')

    def search_cellml(self, query, context=[], topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        return self.__search_object(query, context, topk, min_sim, c_weight, obj_type='cellml')

    def search_all(self, query, context=[], topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
        return {
            'exposure': self.__search_object(query, context, topk, min_sim, c_weight, obj_type='exposure'),
            'workspace': self.__search_object(query, context, topk, min_sim, c_weight, obj_type='workspace'),
            'cellml': self.__search_object(query, context, topk, min_sim, c_weight, obj_type='exposure')
        }
    
    def search_by_cellml(self, query, context=[], topk=TOP_K, min_sim=MIN_SIM, c_weight=C_WEIGHT):
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
        for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
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
    
#===============================================================================
