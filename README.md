# sckan_pmr_search

This package is designed to search for models in the Physiome Repository Model (PMR). It has two main functionalities: 1. generating a mapping of SCKAN terms to CellML files in the PMR, and 2. being reused as a package to search for any terms or text within Python code.

## SCKAN2PMR Generator

This functionality is intended to generate a mapping of SCKAN terms to models or CellML files in the PMR. The results are stored as json files within a single zip file.

### Installation and SCKAN2PMR Generation

1. For SCKAN2PMR generation purpose, we use poetry environment. Follow this [link](https://python-poetry.org/) to install.
2. Clone this repository:
   ```
    git clone git@github.com:napakalas/sckan-pmr-search.git
   ```
3. Navigate to the main directory.
4. Checkout to the the latest release. The list of releases can be found [here](https://github.com/napakalas/sckan-pmr-search/releases). For example, if the latest release is `v0.0.2`, then the command is:
   ```
    git checkout tags/v0.0.2
   ```
5. Install using poetry:
   ```
    poetry install
   ```
6. Generate SCKAN2PMR mapping file:
   ```
    poetry run python run_sckan2pmr.py --exposure-only
   ```
7. The result is a zip file named `release-sckan2pmr-x.x.x.zip`, where `x.x.x` is the release tag, located in `./pmrsearch/output/sckan2pmr-releases/` directory. For the `v0.0.2` example above, the file name becomes `release-sckan2pmr-0.0.2.zip`. This zip file contains four files:
   * `sckan2pmr.json`: Mapping of SCKAN terms to CellML files.
   * `sckan2pmr.schema.json`: JSON schema defining the data structure of the `sckan2pmr.json` file.
   * `metadata.json`: Metadata describing the SCKAN source, pmr2sckan version, etc.
   * `metadata.schema.json`: JSON schema defining the data structure of the `metadata.json` file.

### Properties for SCKAN2PMR Generation

Note that all properties are optional. It is advisable to use `--exposure-only` to filter the results by retrieving CellML files related to an exposure. The list of properties are:

* `--min-sim` : Minimum similarity is used as the retrieval threshold. The default value is 0.7.
* `--exposure-only` : Filter the results to retrieve exposure only.
* `--workspace` : Folder of the downloaded workspaces.
* `--rebuild-index` : Generate a new SCKAN and CellML index before generating the SCKAN2PMR mapping.
* `--clean-extraction` : This option is in addition to `--rebuild-index`, incorporating SCKAN and PMR updates and extracts.
* `--sckan-version` : Specify the SCKAN version to build indexes.
Running the script with `--build-index` without `--clean-extraction` will look for any updates in SCKAN and the PMR. Any additional updates will be indexed into the current indexes. However, using `--clean-extraction` means that all indexes will be rebuilt along with the knowledge in SCKAN and the PMR. Not specifying `--sckan-version` means that indexes will use the current index SCKAN version or the newest SCKAN version.

## Search Package

This package can be used as a searching module in a Python program. The query can be a SCKAN term ID such as `UBERON:0003645` or free text such as `metacarpal bone of digit 1`. Furthermore, the query can be equipped with `context` information, minimum similarity (`min_sim`), and context weighting (`c_weight`). There are four main methods, each with the same arguments:

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

searcher.search_exposure(query='UBERON:0001283', topk=5,  min_sim=0.8, c_weight=0.8)
# result::
# [
#     (0.8123948574066162, 'https://models.physiomeproject.org/exposure/5eb53afcc123269dab6d7c4299814a18'),
# ]

searcher.search_cellml('UBERON:0001283')
# results:
# [
#     (0.8123948574066162, 'https://models.physiomeproject.org/workspace/zager_schlosser_tran_2007/rawfile/248df9f7c5072e210267eb2b3a09b51606ea364c/zager_schlosser_tran_2007.cellml'), 
#     (0.7802648544311523, 'https://models.physiomeproject.org/workspace/tolic_mosekilde_sturis_2000/rawfile/e3fea860d1d39726b9a239357a5ccbdaee4b3910/tolic_mosekilde_sturis_2000.cellml'), 
#     (0.7216819524765015, 'https://models.physiomeproject.org/workspace/297/rawfile/5c6d2c520a9faad399fd954401d658362d25ef03/AE1/Weinstein_2000_AE1_Fig2E.cellml'),
#     ...
# ]

searcher.search_workspace('bile canaliculus', 'Liver')
# results:
# [
#     (0.8233466148376465, 'https://models.physiomeproject.org/workspace/tolic_mosekilde_sturis_2000'), 
#     (0.784793496131897, 'https://models.physiomeproject.org/workspace/zager_schlosser_tran_2007'), 
#     (0.7037129402160645, 'https://models.physiomeproject.org/workspace/297'),
#     ...
# ]

searcher.search_all('bile canaliculus', 'Liver')
# results:
# {
#     'exposure': [(0.8233466148376465, 'https://models.physiomeproject.org/exposure/af45c0110f7e0c42bb5904aebdcf37da'), ...], 
#     'workspace': [(0.8233466148376465, 'https://models.physiomeproject.org/workspace/tolic_mosekilde_sturis_2000'), ...], 
#     'cellml': [(0.8233466148376465, 'https://models.physiomeproject.org/workspace/tolic_mosekilde_sturis_2000/rawfile/e3fea860d1d39726b9a239357a5ccbdaee4b3910/tolic_mosekilde_sturis_2000.cellml'), ...]
#     }
```
