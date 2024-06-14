import torch.backends.mps
from pmrsearch.indexer.clusterer import CellmlClusterer
from pmrsearch.setup import SEARCH_FILE, WORKSPACE_DIR
from pprint import pprint
import torch

#===============================================================================

def main():
    device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    map_location = torch.device(device)
    data = torch.load(SEARCH_FILE, map_location=map_location)

    print('Create clusterer with XPath and structure features using HDBSCAN')
    pmr_clusterer = CellmlClusterer(workspace_dir=WORKSPACE_DIR, cellmls=data['cellml'])

    pprint(pmr_clusterer.getDict())

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# poetry run python tools/build_clusterer.py --workspace ../pmrsearch/workspaces/
