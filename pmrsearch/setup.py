
import os
import json
import re

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCE_PATH = f'{CURRENT_PATH}/resources'

SEARCH_FILE = f'{CURRENT_PATH}/indexes/search_data.pt'
SCKAN_BERT_FILE = f'{CURRENT_PATH}/indexes/sckan_bert.pt'
SCKAN_FILE = f'{CURRENT_PATH}/indexes/sckan.pt'

SCKAN_GRAPH = f'{RESOURCE_PATH}/sckan.graph' 
SCKAN_TERMS = f'{RESOURCE_PATH}/sckan_terms.json'

SCKAN2PMR =  f'{CURRENT_PATH}/output/sckan2pmr.json'
SCKAN2PMR_SQLITE = f'{CURRENT_PATH}/output/sckan2pmr.db'

PMR_URL = 'https://models.physiomeproject.org/'

BERTModel = 'multi-qa-MiniLM-L6-cos-v1'
BIOBERT = 'gsarti/biobert-nli'
NLPModel = 'en_core_sci_scibert'

METADATA_FILE = f'{CURRENT_PATH}/output/metadata.json'
with open(METADATA_FILE, 'r') as f:
    METADATA = json.load(f)

def url_to_curie(url):
    if url.startswith('http'):
        class_id = url.upper().rsplit('/', 1)[-1].rsplit('#', 1)[-1].replace('_', ':')
        if ':' not in class_id:
            match = re.match(r"([a-zA-Z]+)([0-9]+)", class_id)
            try:
                class_id = match.group(1) + ':' + match.group(2)
            except Exception:
                return class_id
    elif url.startswith('urn:miriam'):
        class_id = url.rsplit(':', 1)[-1].replace('%3A', ':')
    else:
        return url
    return class_id
