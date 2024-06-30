import requests
from bs4 import BeautifulSoup
from sympy import exp_polar
from pmrsearch.setup import LOOKUP_TIMEOUT, WORKSPACE_DIR, PMR_URL, EXPOSURES_FILE, getJsonFromPmr, loadJson, getUrlFromPmr
from pmrsearch.indexer import RS_WORKSPACE
import re
from pprint import pprint
from tqdm import tqdm
import json
import git


#===============================================================================
## load workspaces
workspaces = loadJson(RS_WORKSPACE)

## functions for authors and description update
valid_email_regex = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
bot_list = ['nobody@models.cellml.org', 'noreply']

def update_authors_and_created_date(exposure):
    if (workspace:=workspaces.get(exposure['workspace'])) is not None:
        work_dir = f"{WORKSPACE_DIR}/{workspace['workingDir']}"
        repo = git.Repo(work_dir)
        commits = list(repo.iter_commits())
        commits.reverse()
        for commit in commits:
            authors = []
            # check author, doesn't include bot
            if not (any(bot in str(commit.author.email) for bot in bot_list) or not re.search(valid_email_regex, str(commit.author.email))):
                for author in authors:
                    if commit.author.name in author or commit.author.email in author:
                        authors.remove(author)
                authors = [f'{commit.author.name} <{commit.author.email}>'] + authors
            if commit.hexsha == exposure['sha']:
                for author in authors:
                    if exposure['authors'].split('<')[0] in author or exposure['authors'].split('<')[-1] in author:
                        authors.remove(author)
                if len(authors) == 0:
                    authors.append(exposure['authors'])
                elif exposure['authors'] != 'admin':        
                    authors = [exposure['authors']] + authors
                exposure['authors'] = ', '.join(authors)
                exposure['created'] = str(commit.committed_datetime)
                return

def update_description(exposure):
    if (workspace:=workspaces.get(exposure['workspace'])) is not None:
        if workspace['description'] is not None:
            if len(exposure['description']) == 0 and len(workspace['description']) > 0:
                exposure['description'] = workspace['description']

## a function to extract exposure
def extract_exposure(exposure_url):
    based_info = getJsonFromPmr(exposure_url)
    exposure_info = {'exposure': exposure_url}
    for item in based_info.get('items', []):
        for data in item.get('data'):
            if data['name'] == 'title':
                exposure_info['title'] = data['value']
            if data['name'] == 'commit_id':
                exposure_info['sha'] = data['value']
    for link in based_info.get('links', []):
        if link['prompt'] == 'Workspace URL':
            exposure_info['workspace'] = link['href']
            break

    r = requests.get(exposure_url, timeout=LOOKUP_TIMEOUT)

    soup = BeautifulSoup(r.content, 'html.parser')
    # make sure that workspace and sha is available, if not get it from soup
    if 'workspace' not in exposure_info or 'sha' not in exposure_info:
        if len(ahref:=soup.find_all('dl', {'id':'pmr_source'})) > 0:
            wks = ahref[0].find_all('a')
            exposure_info['workspace'] = wks[0]['href']
            exposure_info['sha'] = wks[1]['href'].split('/')[-2]
    # return if cannot find workspace
    if 'workspace' not in exposure_info:
        return [exposure_info]
    
    if len(omexes:=soup.find_all('a', href=re.compile('omex'))):
        exposure_info['omex'] = omexes[0]['href']
    for image in soup.find_all('img'):
        if 'logo-physiome.png' not in (src:=image['src']):
            # print(src)
            if not src.startswith('http'):
                exposure_info['image'] = f"{exposure_info['workspace']}/@@rawfile/{exposure_info['sha']}/{src}"
            else:
                exposure_info['image'] = src
            break
    # info from workspace
    exposure_info['authors'] = []
    exposure_info['description'] = ''
    if exposure_info['workspace'] not in workspaces:
        exposure_info['authors'] = ', '.join(list(set(exposure_info['authors'])))
        return [exposure_info]
    try:
        r = getJsonFromPmr(exposure_info['workspace'])
        for item in r['items']:
            for data in item['data']:
                if data['name'] == 'owner':
                    exposure_info['authors'] += [data['value']]
                if data['name'] == 'description':
                    if data['value'] is not None:
                        exposure_info['description'] = data['value']
                    update_description(exposure_info)
    except:
        update_description(exposure_info)

    exposure_info['authors'] = ', '.join(list(set(exposure_info['authors'])))
    update_authors_and_created_date(exposure_info)
    return [exposure_info]

#===============================================================================

exposure_data = loadJson(EXPOSURES_FILE)
available_exposure = {e['exposure']:e for e in exposure_data}
exposure_infos = []
for exposure_url in tqdm(exposure_urls:=getUrlFromPmr(PMR_URL + 'exposure')):
    if exposure_url in available_exposure:
        if 'authors' in available_exposure[exposure_url]: # temporary
            exposure_infos += [available_exposure[exposure_url]]
        else:
            exposure_infos += extract_exposure(exposure_url)
    else:
        exposure_infos += extract_exposure(exposure_url)

with open('/Users/ymun794/Documents/MapCore/fc-pmr-search/pmrsearch/output/exposures2.json', 'w') as f:
    json.dump(exposure_infos, f, indent=4)

        