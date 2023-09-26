
from pmrsearch.indexer import sckan_crawling

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extracting information from SCKAN")
    parser.add_argument('--sckan', dest='sckanURL', help='A URL to SCKAN release')
    parser.add_argument('--dest', dest='destFile', help='Destination file to store rdf pickle SCKAN')
    inputs = vars(parser.parse_args())

    
    
    
#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================