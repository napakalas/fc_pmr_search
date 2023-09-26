#===============================================================================

import argparse
from pmrsearch import PMRSearcher

#===============================================================================

def main():
    parser = argparse.ArgumentParser(description='Searching for cellml model in the PMR.')
    parser.add_argument('--query', dest='query',
                        help='query term or ontology terms')
    parser.add_argument('--return-type', dest='querytype', default='exposure',
                        help='the type of of return document query, (exposure|workspace|cellml|all), default="exposure"')
    args = parser.parse_args()

    model = PMRSearcher()
    if args.querytype == 'all':
        print(model.search_all(args.query))
    elif args.querytype == 'exposure':
        print(model.search_exposure(args.query))
    elif args.querytype == 'workspace':
        print(model.search_workspace(args.query))
    elif args.querytype == 'cellml':
        print(model.search_by_cellml(args.query))

if __name__ == '__main__':
    main()