
from pmrsearch.searcher import PMRSearcher

#===============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Creating search index for SCKAN and PMR")
    parser.add_argument('--min-sim', dest='minimalSimilarity', help='Minimal similarity.', default=0.7)
    parser.add_argument('--exposure-only', dest='exposureOnly', help='Returning cellml with exposure only', action='store_true')
    args = parser.parse_args()

    searcher = PMRSearcher()
    searcher.generate_and_save_sckan2pmr(min_sim=args.minimalSimilarity, exposure_only=args.exposureOnly)

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# Running script:
# python create_sckan2pmr.py
# python create_sckan2pmr.py --min-sim 0.7
# python create_sckan2pmr.py --exposure-only
# python create_sckan2pmr.py --exposure-only --min-sim 0.7
