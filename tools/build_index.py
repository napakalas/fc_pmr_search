
from pmrsearch.indexer import PMRIndexer

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Creating search index for SCKAN and PMR")
    parser.add_argument('--workspace', dest='workspaceDir', help='Path to PMR workspaces')
    parser.add_argument('--clean-extraction', dest='cleanExtraction', help='Clean extraction will crawl all knowledges', action='store_true')
    args = parser.parse_args()

    pmr_indexer = PMRIndexer(pmr_workspace_dir=args.workspaceDir)

    pmr_indexer.create_search_index(clean_extraction=args.cleanExtraction)
    
    
#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# poetry run python tools/build_index.py
# poetry run python tools/build_index.py --clean-extraction
# poetry run python tools/build_index.py --workspace ../pmrsearch/workspaces
# poetry run python tools/build_index.py --workspace ../pmrsearch/workspaces --clean-extraction
