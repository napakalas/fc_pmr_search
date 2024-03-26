
from pmrsearch.indexer import PMRIndexer

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Creating search index for SCKAN and PMR")
    parser.add_argument('--workspace', dest='workspaceDir', help='Path to PMR workspaces')
    args = parser.parse_args()

    pmr_indexer = PMRIndexer(pmr_workspace_dir=args.workspaceDir)

    pmr_indexer.create_search_index()
    
    
#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# poetry run python tools/build_index.py --workspace /Users/ymun794/Documents/Git_PhD/casbert-indexer/casbert_indexer/workspaces
