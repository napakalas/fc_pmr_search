from pmrsearch.setup import SCKAN2PMR, CURRENT_PATH
import json
import os

EVALUATION_FILE = f'{CURRENT_PATH}/output/evaluation_results.json'

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Creating search index for SCKAN and PMR")
    parser.add_argument('--annotation', dest='annotations', help='Path FC annotation json file', default=None)
    args = parser.parse_args()

    # data = torch.load(SEARCH_FILE, map_location=torch.device('cpu'))
    # sckan_term = data['sckanTerm']

    with open(SCKAN2PMR, 'r') as f:
        sckan2pmr = json.load(f)

    missed = {}
    found = {}
    included = ['FTU Name', 'Organ', 'Model', 'Label', 'Nerve Name', 'Organ/System', 'Models', 'Organ Name', 'Systems', 'System Name', 'Vessel Name']
    if args.annotations is None:
        pass
    elif os.path.exists(args.annotations):
        with open(args.annotations, 'r') as f:
            annotations = json.load(f)
        for values in annotations.values():
            for value in values:
                new_value = {k:v for k, v in value.items() if k in included}
                term_id = value.get('Model', value.get('Models', ''))
                if term_id not in sckan2pmr:
                    missed[term_id] = [new_value]
                else:
                    new_value['results'] = sckan2pmr[term_id]
                    found[term_id] = [new_value]

    with open(EVALUATION_FILE, 'w') as f:
        json.dump({'found':found, 'missed':missed}, f, indent=4)
    
    print('Found:', len(found), '; missed:', len(missed))
    
#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================

# Running script:
# python create_sckan2pmr.py --annotation 'Path FC annotation json file'
    