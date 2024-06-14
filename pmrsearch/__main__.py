#===============================================================================

import argparse
import os
import zipfile

from pmrsearch import PMRSearcher
from pmrsearch.indexer.pmr_crawler import Workspaces
from pmrsearch.indexer import PMRIndexer, RS_WORKSPACE
from pmrsearch.setup import WORKSPACE_DIR, METADATA_FILE, SCKAN2PMR, SCHEMA_METADATA_FILE, SCHEMA_SCKAN2PMR_FILE, CURRENT_PATH
from pmrsearch.setup import loadJson

#===============================================================================

def main():
    parser = argparse.ArgumentParser(description='Searching for cellml model in the PMR.')
    parser.add_argument('--min-sim', dest='minimalSimilarity', help='Minimal similarity.', default=0.7)
    parser.add_argument('--exposure-only', dest='exposureOnly', help='Returning cellml with exposure only', action='store_true')
    parser.add_argument('--workspace', dest='workspaceDir', help='Path to PMR workspaces', default=WORKSPACE_DIR)
    parser.add_argument('--clean-extraction', dest='cleanExtraction', help='Clean extraction will crawl all knowledges', action='store_true')
    parser.add_argument('--rebuild-index', dest='rebuildIndex', help='Generating ', action='store_true')
    args = parser.parse_args()

    if args.rebuildIndex:
        # updating workspaces, checking for any update in the PMR
        workspaces = Workspaces(RS_WORKSPACE)
        workspaces.update()

        # updating SCKAN and create PMR index
        pmr_indexer = PMRIndexer(pmr_workspace_dir=args.workspaceDir)
        pmr_indexer.create_search_index(clean_extraction=args.cleanExtraction)

    # creating SCKAN2PMR mapping
    searcher = PMRSearcher()
    searcher.generate_term_to_pmr_save(min_sim=args.minimalSimilarity, exposure_only=args.exposureOnly)

    # save the generated CKAN2PMR as a zip file
    metadata = loadJson(METADATA_FILE)
    release_path = f'{CURRENT_PATH}/output/sckan2pmr-releases/release-sckan2pmr-{metadata["pmrindexer_version"]}.zip'
    if os.path.exists(release_path):
        os.remove(release_path)
    zf = zipfile.ZipFile(release_path, mode="w")
    for file_path in [METADATA_FILE, SCKAN2PMR, SCHEMA_METADATA_FILE, SCHEMA_SCKAN2PMR_FILE]:
        zf.write(file_path, file_path.split('/')[-1], zipfile.ZIP_DEFLATED)
    zf.close()

if __name__ == '__main__':
    main()

#===============================================================================

# python run_sckan2pmr.py --exposure-only

#===============================================================================
