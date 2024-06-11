import rdflib
import os
from tqdm import tqdm
import logging
import tempfile
import requests
import zipfile
import pickle
import json
from rdflib.namespace import OWL, RDFS
import torch
import logging  as log
import shutil

from ..setup import SCKAN_GRAPH, METADATA, METADATA_FILE, url_to_curie, SCKAN_TERMS, SCKAN_BERT_FILE, SCKAN_BIOBERT_FILE, request_json

NPO_OWNER = 'SciCrunch'
NPO_REPO = 'NIF-Ontology'
NPO_API = f'https://api.github.com/repos/{NPO_OWNER}/{NPO_REPO}'
NPO_RAW = f'https://raw.githubusercontent.com/{NPO_OWNER}/{NPO_REPO}'
NPO_GIT = f'https://github.com/{NPO_OWNER}/{NPO_REPO}'

class NPOException(Exception):
    pass

def __get_ttl(path):
    """
    Returning list of ttl_files and non ttl_files.
    path: a path to the extracted SCKAN containing ttl files.
    """
    ttl_files = []
    other_files = []
    for folder in next(os.walk(path), (None, None, []))[1:]:
        for p in folder:
            np = os.path.join(path, p)
            if os.path.isdir(np):
                tmp =__get_ttl(np)
                ttl_files += tmp[0]
                other_files += tmp[1]
            elif p.endswith('ttl'):
                ttl_files += [np]
            else:
                other_files += [np]
    return [ttl_files, other_files]

def __load_sckan(sckan_path, dest_file):
    # getting all ttl files
    filenames = __get_ttl(sckan_path)[0]
    if len(filenames) == 0:
        logging.warning('No ttl file available.')

    # parsing all ttl files
    g = rdflib.Graph()
    for filename in tqdm(filenames):
        try:
            g.parse(filename)
        except Exception:
            logging.error('Cannot load file: {}'.format(filename))

    if dest_file is not None:
        with open(dest_file, 'wb') as f:
            pickle.dump(g, f)

    return g

def __download_sckan(url):
    temp_dir = tempfile.mkdtemp()
    # Create the full path for the temporary ZIP file
    temp_zip_file_path = os.path.join(temp_dir, "release.zip")
    try:
        # Send an HTTP GET request to the GitHub raw file URL
        response = requests.get(url, timeout=10)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:

            # Write the content of the response to the temporary ZIP file
            with open(temp_zip_file_path, 'wb') as temp_zip_file:
                temp_zip_file.write(response.content)

            # Extract the contents of the ZIP file to the temporary folder
            with zipfile.ZipFile(temp_zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            logging.info(f"ZIP file downloaded and extracted to: {temp_dir}")
            return temp_dir
        else:
            logging.error(f"Failed to download ZIP file. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def extract_sckan_terms(ontologies, to_embedding, bert_model, biobert_model, nlp_model, sckan_version=None, store_as=None, clean_extraction=True):
    """
    ontologies: graph of ontology collections
    to_embedding: a function to calculate embedding from query
    bert_model: bert model to convert terms to embedding
    biobert_model: biobert model to convert terms to embedding
    nlp_model: model to identify named entity, etc
    sckan_version: the version of sckan, located in https://github.com/SciCrunch/NIF-Ontology/releases, e.g. sckan-2024-03-26
    store_as: the crawled file is loaded into rdflib graph and then stored as a file using pickle
    device: adjust based on machine availability, cpu or gpu
    clean_extraction: when True, this will reextract sckan
    """

    if torch.cuda.is_available():
        device = 'gpu'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    if (npo_release:=check_npo_release(sckan_version)) is not None:
        if sckan_version is None:
            sckan_version = npo_release['sckan_version']
    else:
        logging.error('Local SCKAN graph file is not available. URL should be provided')
        return None, None, None

    store_as = SCKAN_GRAPH if store_as is None else store_as
    if not clean_extraction and METADATA.get('sckan_version', '') == sckan_version:
        try:
            if (METADATA.get('sckan_version', '') == sckan_version or sckan_version is None) and os.path.exists(store_as):
                with open(SCKAN_TERMS, 'r') as f:
                    sckan_terms = json.load(f)
                map_location = torch.device(device)
                sckan_bert_embs = torch.load(SCKAN_BERT_FILE, map_location=map_location)
                sckan_biobert_embs = torch.load(SCKAN_BIOBERT_FILE, map_location=map_location)
                return sckan_terms, sckan_bert_embs, sckan_biobert_embs
        except Exception:
            logging.warning('Cannot load the identified graph file. Continue to loading the provided URL')
            return

    # download file
    sckan_url = __download_sckan(npo_release['sckan_url'])
    # load SCKAN
    g = __load_sckan(sckan_url, store_as)
    # update METADATA sckan_url
    for k, v in npo_release.items():
        METADATA[k] = v
    # save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(METADATA, f)
    
    sckan_terms = {}
    for s, p, o in tqdm(g):
        if 'ilx_' in str(s) or 'UBERON' in str(s):
            class_id = url_to_curie(str(s))
            if class_id not in sckan_terms:
                sckan_terms[class_id] = {'label':'', 'def':[], 'synonym':[], 'is_a':[], 'is_a_text':[]}
            # get label / name
            if p == RDFS.label:
                sckan_terms[class_id]['label'] = str(o)
            # get description
            if 'IAO_0000115' in str(p):
                sckan_terms[class_id]['def'] += [str(o)]
            # get parent / is_a
            if p == RDFS.subClassOf:
                is_a = None
                if str(o).startswith('http'):
                    is_a = url_to_curie(str(o))
                else:
                    for _, o2 in g.predicate_objects(o):
                        if str(o2).startswith('http'):
                            is_a = url_to_curie(str(o2))
                if is_a is not None and is_a != OWL.Restriction and is_a not in sckan_terms[class_id]['is_a']:
                    sckan_terms[class_id]['is_a'] += [str(is_a)]
            # synonym
            if 'synonym' in str(p).lower():
                sckan_terms[class_id]['synonym'] += [str(o)]
            # delete if empty
            if sckan_terms[class_id]['label'] == '':
                del sckan_terms[class_id]

    # get parent / is_a text
    for _, desc in tqdm(sckan_terms.items()):
        for is_a in desc['is_a']:
            if is_a in sckan_terms:
                if 'label' in sckan_terms[is_a]:
                    desc['is_a_text'] +=  [sckan_terms[is_a]['label']]
                if is_a in ontologies.index:
                    desc['is_a_text'] += [ontologies.loc[is_a]['name']]
        desc['is_a_text'] = list(set(desc['is_a_text']))

    # calculate embeddings
    sckan_bert = {k:to_embedding(v, bert_model, nlp_model) for k, v in tqdm(sckan_terms.items())}
    sckan_bert_embs = {'id': list(sckan_bert.keys()), 'embs': torch.stack(list(sckan_bert.values()))}

    sckan_biobert = {k:to_embedding(v, biobert_model, nlp_model) for k, v in tqdm(sckan_terms.items())}
    sckan_biobert_embs = {'id': list(sckan_biobert.keys()), 'embs': torch.stack(list(sckan_biobert.values()))}

    # save to files
    with open(SCKAN_TERMS, 'w') as f:
        json.dump(sckan_terms, f)
    torch.save(sckan_bert_embs, SCKAN_BERT_FILE)
    torch.save(sckan_biobert_embs, SCKAN_BIOBERT_FILE)

    # Clean up: Delete the temporary ZIP file and its parent directory
    if sckan_url is not None:
        shutil.rmtree(sckan_url, ignore_errors=True)

    return sckan_terms, sckan_bert_embs, sckan_biobert_embs

def check_npo_release(npo_release) -> dict:
    #=================================================
        if (response:=request_json(f'{NPO_API}/releases')) is not None:
            releases = {r['tag_name']:r for r in response if r['tag_name'].startswith('sckan-')}
            if npo_release is None:
                if len(releases):
                    # Use most recent
                    npo_release = sorted(releases.keys())[-1]
                    log.warning(f'No NPO release given: used {npo_release}')
                else:
                    raise NPOException('No NPO releases available')
            elif npo_release not in releases:
                raise NPOException(f'Unknown NPO release: {npo_release}')

            release = releases[npo_release]
            response = request_json(f'{NPO_API}/git/refs/tags/{release["tag_name"]}')
            browser_download_url = ''
            for asset in release['assets']:
                if 'release' in asset['browser_download_url'].split('/')[-1]:
                    browser_download_url = asset['browser_download_url']
                    break
            npo_build = {
                'sckan_sha': response['object']['sha'] if response is not None else None,
                'sckan_build': release['created_at'].split('T')[0],
                'sckan_version': release["tag_name"],
                'sckan_path': f'{NPO_GIT}/tree/{release["tag_name"]}',
                'sckan_url': browser_download_url
            }
            return npo_build
        else:
            raise NPOException(f'NPO at {NPO_API} is not available')

#===============================================================================
