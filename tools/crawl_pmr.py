from pmrsearch.indexer.pmr_crawler import Workspaces
from pmrsearch.indexer import RS_WORKSPACE

#===============================================================================

workspaces = Workspaces(RS_WORKSPACE)
workspaces.update()

#===============================================================================
