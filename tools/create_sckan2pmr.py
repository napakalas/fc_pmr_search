
from pmrsearch.searcher import PMRSearcher

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Creating search index for SCKAN and PMR")
    parser.add_argument('--annotation', dest='annotations', help='Path FC annotation json file', default=None)
    args = parser.parse_args()

    searcher = PMRSearcher()
    searcher.generate_term_to_pmr_save(min_sim=0.7)
    searcher.check_and_generate_annotation_completeness(args.annotations)

    
    
#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# Running script:
# python create_sckan2pmr.py --annotation directory_to_store
    