from pmrsearch import PMRSearcher

#===============================================================================

def main():
    query = "UBERON:0000002"
    model = PMRSearcher()
    
    print(f'The results of search_all method for query {query}')
    print(model.search_all(query))
    print(f'The results of search_exposure method for query {query}')
    print(model.search_exposure(query))
    print(f'The results of search_workspacve method for query {query}')
    print(model.search_workspace(query))
    print(f'The results of search_by_cellml method for query {query}')
    print(model.search_by_cellml(query))

if __name__ == '__main__':
    main()

#===============================================================================
