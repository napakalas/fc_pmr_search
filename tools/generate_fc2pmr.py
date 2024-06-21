import json

import requests
from pmrsearch.searcher import PMRSearcher
from pmrsearch.setup import WORKSPACE_DIR, LOOKUP_TIMEOUT
import logging as log

#===============================================================================

FC_URL = 'https://github.com/AnatomicMaps/functional-connectivity'
FC_GIT_API_REFS = 'https://api.github.com/repos/AnatomicMaps/functional-connectivity/git/refs'
INCLUDES = ['FTU Name', 'Organ', 'Model', 'Label', 'Nerve Name', 'Organ/System', 'Models', 'Organ Name', 'Systems', 'System Name', 'Vessel Name']
#===============================================================================

def check_sha(sha, api_refs):
    r = requests.get(api_refs, timeout=LOOKUP_TIMEOUT)
    candidate_sha = r.json()[0].get('object').get('sha')
    if sha is None:
        log.warning(f'SHA is not provided, using {candidate_sha} instead')
        return candidate_sha
    for r_sha in r.json():
        if (f_sha:=r_sha.get('object').get('sha')).startswith(sha):
            return f_sha
    log.warning(f'SHA {sha} is not found, using {candidate_sha} instead')
    return candidate_sha
    

def get_fc_terms(sha):
    # get annotation data
    url = f'{FC_URL}/raw/{sha}/annotation.json'
    r = requests.get(url, timeout=LOOKUP_TIMEOUT)
    annotations = json.loads(r.text)

    terms = set()
    for values in annotations.values():
        for value in values:
            new_value = {k:v for k, v in value.items() if k in INCLUDES}
            term_id = value.get('Model', value.get('Models', ''))
            if term_id is not None or term_id != '':
                terms.add(term_id)
    return terms

def mapping_creation(fc_sha, ac_fhas=[]):
    terms = get_fc_terms(fc_sha)
    searcher = PMRSearcher()
    
    ### check the terms here

    
    from pprint import pprint
    pprint(terms)
            

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Creating search index for Flatmap to PMR")
    parser.add_argument('--fc-sha', dest='fc_sha', help='SHA of a particular FC release')
    parser.add_argument('--rat-sha', dest='fc_sha', help='SHA of a particular AC rat release')
    parser.add_argument('--mouse-sha', dest='fc_sha', help='SHA of a particular AC mouse release')
    parser.add_argument('--human-sha', dest='fc_sha', help='SHA of a particular AC human release')
    parser.add_argument('--cat-sha', dest='fc_sha', help='SHA of a particular AC cat release')
    parser.add_argument('--pig-sha', dest='fc_sha', help='SHA of a particular AC pig release')

    args = parser.parse_args()
    fc_sha = check_sha(args.fc_sha, FC_GIT_API_REFS)

    mapping_creation(fc_sha=fc_sha)

if __name__ == '__main__':
    main()

#===============================================================================
