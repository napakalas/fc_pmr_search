
from pmrsearch.searcher import PMRSearcher

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Creating search index for SCKAN and PMR")
    parser.add_argument('--min-sim', dest='minimalSimilarity', help='Minimal similarity.', default=0.7)
    args = parser.parse_args()

    searcher = PMRSearcher()
    searcher.generate_term_to_pmr_save(min_sim=args.min_sim)

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# Running script:
# python create_sckan2pmr.py --min-sim 'Minimal similarity'
    