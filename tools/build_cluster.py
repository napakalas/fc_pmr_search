
from pmrsearch.indexer.clusterer import CellmlClusterer
import json

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clustering PMR")
    parser.add_argument('--workspace', dest='workspaceDir', help='Path to PMR workspaces')
    args = parser.parse_args()

    
    with open('/Users/ymun794/Documents/MapCore/fc-pmr-search/pmrsearch/resources/listOfCellml.json', 'r') as f:
        cellmls = json.load(f)
    
    print('Create clusterer with XPath and structure features using HDBSCAN')
    pmr_clusterer = CellmlClusterer(workspace_dir=args.workspaceDir, cellmls=cellmls)

    with open('/Users/ymun794/Documents/MapCore/fc-pmr-search/pmrsearch/resources/cellmlClusterer.json', 'w') as f:
        json.dump(pmr_clusterer.getDict(), f, indent=4)

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# poetry run python tools/build_clusterer.py --workspace /Users/ymun794/Documents/MapCore/pmr-search/pmr_search/workspaces
