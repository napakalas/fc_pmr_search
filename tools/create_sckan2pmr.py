
from pmrsearch.searcher import PMRSearcher

def main():
    searcher = PMRSearcher()
    searcher.generate_term_to_pmr_save(min_sim=0.7)
    
    
#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================