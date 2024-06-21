
import logging as log
import os
import json
import re
import requests
import gzip
import pickle
from json import JSONDecodeError

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCE_PATH = f'{CURRENT_PATH}/resources'
WORKSPACE_DIR = f'{CURRENT_PATH}/workspaces'

SEARCH_FILE = f'{CURRENT_PATH}/indexes/search_data.pt'
SCKAN_BERT_FILE = f'{CURRENT_PATH}/indexes/sckan_bert.pt'
SCKAN_BIOBERT_FILE = f'{CURRENT_PATH}/indexes/sckan_biobert.pt'
SCKAN_TERMS = f'{CURRENT_PATH}/indexes/sckan_terms.json'

SCKAN2PMR =  f'{CURRENT_PATH}/output/sckan2pmr.json'

SCHEMA_METADATA_FILE = f'{RESOURCE_PATH}/metadata.schema.json'
SCHEMA_SCKAN2PMR_FILE = f'{RESOURCE_PATH}/sckan2pmr.schema.json'
SCHEMA_EXPOSURES_FILE = f'{RESOURCE_PATH}/exposures.schema.json'

EXPOSURES_FILE = f'{CURRENT_PATH}/output/exposures.json'

PMR_URL = 'https://models.physiomeproject.org/'

BERTModel = 'multi-qa-mpnet-base-dot-v1'
BIOBERT = 'FremyCompany/BioLORD-2023' # this model accommodate semantic textual similarity
NLPModel = 'en_core_sci_lg'

LOOKUP_TIMEOUT = 30

METADATA_FILE = f'{CURRENT_PATH}/output/metadata.json'
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        METADATA = json.load(f)
else:
    METADATA = {}

def url_to_curie(url):
    if url.startswith('http'):
        class_id = url.upper().rsplit('/', 1)[-1].rsplit('#', 1)[-1].replace('_', ':')
        if ':' not in class_id:
            match = re.match(r"([a-zA-Z]+)([0-9]+)", class_id)
            try:
                class_id = match.group(1) + ':' + match.group(2)
            except Exception as e:
                pass
        return class_id
    elif url.startswith('urn:miriam'):
        return url.rsplit(':', 1)[-1].replace('%3A', ':')
    return url

def loadJson(*paths):
    file = os.path.join(CURRENT_PATH, *paths)
    isExist = os.path.exists(file)
    if isExist:
        with open(file, 'r') as fp:
            data = json.load(fp)
        fp.close()
        return data
    else:
        return {}


def dumpJson(data, *paths):
    file = os.path.join(CURRENT_PATH, *paths)
    with open(file, 'w') as fp:
        json.dump(data, fp, indent=4)
    fp.close()

def getJsonFromPmr(url):
    try:
        r = requests.get(url, headers={"Accept": "application/vnd.physiome.pmr2.json.1"}, timeout=LOOKUP_TIMEOUT)
        return r.json()['collection']
    except Exception as e:
        log.error(f'{url} is unreachable')
        return {}

def getAllFilesInDir(*paths):
    drc = os.path.join(CURRENT_PATH, *paths)
    lst = []
    for path, subdirs, files in os.walk(drc):
        for name in files:
            lst += [os.path.join(path, name)]
    return lst

# get list of URLs inside a particulat URL in the PMR
def getUrlFromPmr(url):
    r = requests.get(url, headers={"Accept": "application/vnd.physiome.pmr2.json.1"}, timeout=LOOKUP_TIMEOUT)
    urls = [link['href'] for link in r.json()['collection']['links']]
    return urls

def dumpPickle(data, *paths):
    filename = os.path.join(CURRENT_PATH, *paths)
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

def loadPickle(*paths):
    filename = os.path.join(CURRENT_PATH, *paths)
    file = gzip.GzipFile(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def request_json(endpoint, **kwds):
    try:
        response = requests.get(endpoint,
                                headers={'Accept': 'application/json'},
                                timeout=LOOKUP_TIMEOUT,
                                **kwds)
        if response.status_code == requests.codes.ok:
            try:
                return response.json()
            except JSONDecodeError:
                error = 'Invalid JSON returned'
        else:
            error = response.reason
    except requests.exceptions.RequestException as exception:
        error = f'Exception: {exception}'
    log.warning(f"Couldn't access {endpoint}: {error}")
    return None
