
from pmrsearch.indexer.clusterer import CellmlClusterer
from pmrsearch.setup import SEARCH_FILE, WORKSPACE_DIR
from pprint import pprint
import torch

def main():
    map_location = torch.device('mps')
    data = torch.load(SEARCH_FILE, map_location=map_location)

    print('Create clusterer with XPath and structure features using HDBSCAN')
    pmr_clusterer = CellmlClusterer(workspace_dir=WORKSPACE_DIR, cellmls=data['cellml'])

    pprint(pmr_clusterer.getDict())

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# poetry run python tools/build_clusterer.py --workspace ../pmrsearch/workspaces/
