# fc_pmr_search

## Searcher

This is a package to search for models in the Physiome repository Model (PMR). The query can be a SCKAN term id such as `UBERON:0003645` or free text such as `metacarpal bone of digit 1`. Furthermore, the query can be equipped with `context` information, minimum similarity (`min_sim`), and context weighting (`c_weight`). There are four main methods, with the same arguments:

* search_exposure
* search_workspace
* search_cellml
* search_all

There are two additional methods for debugging purpose:

* search
* search_by_cellml

Example:

```
from pmrsearch import PMRSearcher

searcher = PMRSearcher()

searcher.search_exposure(query='UBERON:0001283', context=['Liver'], topk=5,  min_sim=0.8, c_weight=0.8)
# result::
# [
#     (0.9019607305526733, 'exposure/5eb53afcc123269dab6d7c4299814a18'), 
#     (0.8064868450164795, 'exposure/05510be013e1d096e26c3716c950712d'), 
#     (0.8064868450164795, 'exposure/959a4da5d8a0638f4b36adf5800f4fc0'), 
#     (0.8064868450164795, 'e/9e'), 
#     (0.8064868450164795, 'exposure/c131babb70e87d6f9e66f2a1b1ec861e'), 
# ]

searcher.search_cellml('UBERON:0001283')
# results:
# [
#     (0.8803054094314575, 'workspace/zager_schlosser_tran_2007/rawfile/HEAD/zager_schlosser_tran_2007.cellml'), 
#     (0.7799185514450073, 'workspace/weinstein_2000/rawfile/HEAD/Weinstein_2000_AE1.cellml'), 
#     (0.7799185514450073, 'workspace/584/rawfile/HEAD/weinstein_1998.cellml'),
#     ...
# ]

searcher.search_workspace('bile canaliculus', 'Liver')
# results:
[
    (0.9019607305526733, 'workspace/zager_schlosser_tran_2007'), 
    (0.8064868450164795, 'workspace/potter_zager_barton_2006'), 
    (0.8064868450164795, 'workspace/beard_2005'),
    ...
]

searcher.search_all('bile canaliculus', 'Liver')
# results:
{
    'exposure': [(0.9019607305526733, 'exposure/5eb53afcc123269dab6d7c4299814a18'), ...], 
    'workspace': [(0.9019607305526733, 'workspace/zager_schlosser_tran_2007'), ...], 
    'cellml': [(0.9019607305526733, 'exposure/5eb53afcc123269dab6d7c4299814a18'), ...]
    }
```
## Indexer
