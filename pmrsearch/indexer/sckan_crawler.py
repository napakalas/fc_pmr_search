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

from ..setup import SCKAN_GRAPH, METADATA, METADATA_FILE, url_to_curie, SCKAN_TERMS, SCKAN_BERT_FILE, SCKAN_BIOBERT_FILE

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
    try:
        # Send an HTTP GET request to the GitHub raw file URL
        response = requests.get(url, timeout=10)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Create the full path for the temporary ZIP file
            temp_zip_file_path = os.path.join(temp_dir, "release.zip")

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
    finally:
        # Clean up: Delete the temporary ZIP file and its parent directory
        os.remove(temp_zip_file_path)
        os.rmdir(temp_dir)    

def extract_sckan_terms(ontologies, to_embedding, bert_model, biobert_model, nlp_model, url=None, store_as=None, device='cpu', clean_extraction=True):
    """
    url: a URL †o SCKAN release zip, e.g. https://github.com/SciCrunch/NIF-Ontology/releases/download/sckan-2023-08-04/release-2023-08-04T005709Z-sckan.zip
    ontologies: graph of ontology collections
    embedding_function: a function to calculate embedding from query
    bert_model: bert model to convert terms to embedding
    biobert_model: biobert model to convert terms to embedding
    store_as: the crawled file is loaded into rdflib graph and then stored as a file using pickle
    """

    store_as = SCKAN_GRAPH if store_as is None else store_as
    if not clean_extraction:
        try:
            if (METADATA.get('sckan_url', '') == url or url is None) and os.path.exists(store_as):
                with open(SCKAN_TERMS, 'r') as f:
                    sckan_terms = json.load(f)
                map_location = torch.device(device)
                sckan_bert_embs = torch.load(SCKAN_BERT_FILE, map_location=map_location)
                sckan_biobert_embs = torch.load(SCKAN_BIOBERT_FILE, map_location=map_location)
                print('123')
                return sckan_terms, sckan_bert_embs, sckan_biobert_embs
        except Exception:
            logging.warning('Cannot loaded the identified graph file. Continue to loading the provided URL')
        return
    
    if url is not None:
        # download file
        sckan_path = __download_sckan(url)
        # load SCKAN
        g = __load_sckan(sckan_path, store_as)

        # update METADATA sckan_url
        METADATA['sckan_url'] = url
        # update METADATA sckan_build
        sbj = rdflib.URIRef('http://uri.interlex.org/tgbugs/uris/readable/build/prov')
        pred = rdflib.URIRef('http://uri.interlex.org/tgbugs/uris/readable/build/date')
        for o in g.objects(sbj, pred):
            sckan_build = str(print(o))
        METADATA['sckan_build'] = sckan_build
        # save metadata
        with open(METADATA_FILE, 'w') as f:
            json.dump(METADATA, f)
    else:
        logging.error('Local SCKAN graph file is not available. URL should be provided')
        return None, None, None

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
            if 'synonym' in str(p.lower()):
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

    return sckan_terms, sckan_bert_embs, sckan_biobert_embs
