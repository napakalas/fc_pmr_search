import sys
import requests
import os
import logging as log
import pandas as pd

from pmrsearch.setup import METADATA_FILE, METADATA, RESOURCE_PATH, LOOKUP_TIMEOUT, dumpJson

log.getLogger().setLevel(log.INFO)

SOURCE_URL = "https://data.bioontology.org/ontologies"
API = "apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
ONTOLOGY_URLS = {
    "FMA": f"{SOURCE_URL}/FMA/download?{API}&download_format=csv",
    "CHEBI": f"{SOURCE_URL}/CHEBI/download?{API}",
    "GO": f"{SOURCE_URL}/GO/download?{API}",
    "OPB": f"{SOURCE_URL}/OPB/download?{API}&download_format=csv",
    "UBERON": f"{SOURCE_URL}/UBERON/download?{API}&download_format=csv"
}

def load_and_save_onto_source(onto_file):
    df = pd.DataFrame()
    if onto_file.endswith('.obo'):
        listData = []
        with open(onto_file) as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                line = line.strip()

                if line == '[Term]':
                    data = {}
                    while True:
                        line = fp.readline().strip()
                        keyVals = line.split(': ', 1)
                        if len(keyVals) == 2:
                            if keyVals[0] not in data:
                                data[keyVals[0]] = keyVals[1]
                            else:
                                data[keyVals[0]] += '|'+keyVals[1]
                        if not line:
                            listData += [data]
                            break
        df = pd.DataFrame(listData)
        return df[['id', 'name', 'def', 'synonym', 'is_a']].set_index('id')
    elif '.csv' in onto_file:
        try:
            df = pd.read_csv(onto_file, sep=',', header=0)
        except:
            df = pd.read_csv(onto_file, sep=',', header=0, compression='gzip')
        df.columns = [x.lower() for x in df.columns]
        def update_id(s):
            if not isinstance(s, tuple) and not isinstance(s, str): return ''
            if isinstance(s, tuple):
                s = s[0]
            s = s[s.rfind('/')+1:]
            s = s[s.rfind('#')+1:].replace('_', ':')
            s = s.replace('fma','FMA:') if s.startswith('fma') else s
            return s
        # get 'id'
        if 'id' not in df.columns and 'class id' in df.columns:
            df = df.rename(columns={'class id': 'id'})
            df['id'] = df.apply(lambda x: update_id(x['id']), axis=1)
        # get 'label'
        if 'name' in df.columns:
            df = df.drop('name', axis=1)
        df = df.rename(columns={'preferred label': 'name'})
        # get definition
        df = df.rename(columns={'definitions': 'def'})
        # get synonym
        if 'synonym' in df.columns and "synonyms" in df.columns:
            df['synonym'] = df['synonyms'].fillna('') + ('|'+df['synonym']).fillna('')
        elif 'synonyms' in df.columns:
            df = df.rename(columns={'synonyms':'synonym'})
        # get is_a
        if 'is_a' not in df.columns and 'parents' in df.columns:
            df = df.rename(columns={'parents': 'is_a',})
            df['is_a'] = df.apply(lambda x: update_id(x['is_a']), axis=1)
        return df[['id', 'name', 'def', 'synonym', 'is_a']].set_index('id')
    
def load_ontologies():
    log.info('Loading ontologies ...')
    onto_meta = METADATA.get('ontology', {})
    onto_meta['source'] = SOURCE_URL
    onto_dict = onto_meta.get('type', {})

    list_onto_df = []

    # check ontology version and download new version
    for onto_type, url in ONTOLOGY_URLS.items():
        log.info(f'Loading {onto_type} at {url}')
        submission_url = f'https://data.bioontology.org/ontologies/{onto_type}/latest_submission?{API}'
        r = requests.get(submission_url, timeout=LOOKUP_TIMEOUT)
        file_path = f"{RESOURCE_PATH}/{onto_type.lower()}.{'csv' if url.endswith('csv') else 'obo'}"
        if r.status_code == 200:
            # check current version
            if (current_version:=r.json().get('version')) != onto_dict.get(onto_type, '') or not os.path.exists(file_path):
                # download and replace ontology file
                response = requests.get(url, timeout=LOOKUP_TIMEOUT)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                # update ontology version at metadata
                onto_dict[onto_type] = current_version
        elif not os.path.exists(file_path):
            log.error(f"Ontology server is unreachable and '{file_path}' is not exist")
            sys.exit()
        list_onto_df += [load_and_save_onto_source(file_path)]
    onto_df = pd.concat(list_onto_df, )
    onto_df = onto_df.groupby(onto_df.index).first()  # delete duplicate
    # update metadata
    onto_meta['type'] = onto_dict
    METADATA['ontology'] = onto_meta
    # save metadata
    dumpJson(METADATA, METADATA_FILE)
    return onto_df

#===============================================================================

if __name__ == '__main__':
    load_ontologies()

#===============================================================================
