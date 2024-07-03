#===============================================================================

import argparse
import os
import sys
import zipfile
import logging as log

from pmrsearch import PMRSearcher
from pmrsearch.indexer.pmr_crawler import Workspaces
from pmrsearch.indexer import PMRIndexer, RS_WORKSPACE
from pmrsearch.setup import METADATA, WORKSPACE_DIR, METADATA_FILE, SCKAN2PMR, SCHEMA_METADATA_FILE, SCHEMA_SCKAN2PMR_FILE, CURRENT_PATH
from pmrsearch.setup import loadJson

#===============================================================================

def main():
    parser = argparse.ArgumentParser(description='Searching for cellml model in the PMR.')
    parser.add_argument('--min-sim', dest='minimalSimilarity', help='Minimal similarity.', default=0.7)
    parser.add_argument('--exposure-only', dest='exposureOnly', help='Returning cellml with exposure only', action='store_true')
    parser.add_argument('--workspace', dest='workspaceDir', help='Path to PMR workspaces', default=WORKSPACE_DIR)
    parser.add_argument('--clean-extraction', dest='cleanExtraction', help='Clean extraction will crawl all knowledges', action='store_true')
    parser.add_argument('--rebuild-index', dest='rebuildIndex', help='Generating ', action='store_true')
    parser.add_argument('--sckan-version', dest='sckanVersion', help='SCKAN version of the regenerated index', default=None)
    args = parser.parse_args()

    if args.rebuildIndex:
        # updating workspaces, checking for any update in the PMR
        workspaces = Workspaces(RS_WORKSPACE)
        workspaces.update()

        # updating SCKAN and create PMR index
        if not args.cleanExtraction and args.sckanVersion is not None and args.sckanVersion != (c_version:=METADATA.get('sckan', {}).get('sckan_version')):
            log.error(f'SCKAN version of the current indexes is {c_version}, while this code run is {args.sckanVersion}. \
                      Rerun with --clean-extraction option')
            sys.exit()
        elif not args.cleanExtraction and args.sckanVersion is None:
            args.sckanVersion = METADATA.get('sckan', {}).get('sckan_version')
        pmr_indexer = PMRIndexer(pmr_workspace_dir=args.workspaceDir, sckan_version=args.sckanVersion)
        pmr_indexer.create_search_index(clean_extraction=args.cleanExtraction)

    # creating SCKAN2PMR mapping
    searcher = PMRSearcher()
    searcher.generate_and_save_sckan2pmr(min_sim=float(args.minimalSimilarity), exposure_only=args.exposureOnly)

#===============================================================================
    
if __name__ == '__main__':
    main()

#===============================================================================

# python run_sckan2pmr.py --exposure-only

#===============================================================================
