from distutils.command import clean
import logging
import git.exc
from pmrsearch.setup import getJsonFromPmr, getAllFilesInDir, getUrlFromPmr, dumpPickle, loadJson, dumpJson as dJson
from pmrsearch.setup import PMR_URL, CURRENT_PATH, WORKSPACE_DIR, RESOURCE_PATH

import git
import os
import rdflib
import copy
from tqdm import tqdm

class PmrCollection:
    def __init__(self, *paths):
        self.dataDict = loadJson(*paths)
        self.paths = paths
        if len(self.dataDict)==0:
            self.dataDict['data'] = {}
        self.statusC = {}
        self.data = self.dataDict['data']

    def getJson(self):
        return self.dataDict

    def dumpJson(self):
        self.dataDict['status'] = self.statusC
        for k, v in self.dataDict['data'].items():
            if isinstance(v,dict):
                self.dataDict['vars'] = list(v.keys())
            break
        dJson(self.dataDict, *self.paths)

    def getStatus(self):
        return self.dataDict['status']

    def getData(self):
        return self.data

    def getObjData(self, id, items=[], isCopy=False):
        if not isCopy:
            return self.data[id]
        if len(items) == 0:
            return copy.deepcopy(self.data[id])
        retObj = {}
        for item in items:
            if item in self.data[id]:
                if item == 'rdfLeaves':
                    retObj[item] = self.getObjLeaves(id)
                else:
                    retObj[item] = self.data[id][item]
        return retObj

    def getObjLeaves(self, id):
        leaves = []
        if id in self.data:
            if 'rdfLeaves' in self.data[id]:
                for leaf in self.data[id]['rdfLeaves']:
                    if not leaf.startswith('file:'):
                        leaves += [leaf]
        return leaves

    def getNewId(self):
        return self.__class__.__name__[:3]+'Id-'+str(len(self.data))

    def addRdf(self, id, rdf, rdfLeaves, cmeta):
        if 'rdf' not in self.data[id]:
            self.data[id]['rdf'] = []
        if 'rdfLeaves' not in self.data[id]:
            self.data[id]['rdfLeaves'] = []
        self.data[id]['rdf'] += rdf
        self.data[id]['rdfLeaves'] += rdfLeaves
        self.data[id]['cmeta'] = cmeta

    def getCMeta(self, id):
        if 'cmeta' in self.data[id]:
            return self.data[id]['cmeta']
        return None

class WorkspaceCollection(PmrCollection):
    def __init__(self, *paths):
        super().__init__(*paths)
        self.allWksDir = os.path.join(WORKSPACE_DIR)
        self.statusC = {'deprecated': 0, 'current': 1, 'validating': 2}

    def getCommit(self, url):
        return self.data.get(url,{}).get('commit', 'HEAD')

class Workspaces(WorkspaceCollection):
    def __init__(self, *paths):
        super().__init__(*paths)

    # get list of workspaces URL in the PMR
    def getListWorkspaces(self, fromServer=False):
        if fromServer:
            return getUrlFromPmr(PMR_URL + 'workspace')
        else:
            return list(self.data.keys())

    def getListExposures(self, fromServer=False):
        if fromServer:
            return getUrlFromPmr(PMR_URL + 'exposure')
        else:
            return set([e for v in self.data.values() for e in v.get('exposures',[])])

    def update(self, clean_update=False):
        listWorkspace = self.getListWorkspaces(fromServer=True)
        if clean_update:
            self.data = {}
        # get local repos
        repo_dirs = [os.path.join(os.path.join(WORKSPACE_DIR,name))
                     for name in os.listdir(WORKSPACE_DIR)
                     if os.path.isdir(os.path.join(WORKSPACE_DIR,name))]
        print('Get local repo info')
        for repo_dir in tqdm(repo_dirs):
            self.__local_repo_info(repo_dir)
        self.dumpJson()
        # check current repo update
        print('Check repo update')
        for url in tqdm(set(listWorkspace) & set(self.data.keys())):
            self.__synchroWorkspace(url)
        self.dumpJson()
        # get or update workspaces
        print('Download new repo')
        for url in tqdm(set(listWorkspace) - set(self.data.keys())):
            self.__cloneWorkspace(url)
        self.dumpJson()
        # removed those unidentified workspace
        for url in (set(self.data.keys()) - set(listWorkspace)):
            del self.data[url]
        # Synchronise with exposure
        print('Synchronise with exposure')
        self.__get_exposures()
        self.dumpJson()
        # Now update workspace level RDF
        self.__updateRdf()

    def __local_repo_info(self, repo_dir, repo=None):
        repo = repo if repo is not None else git.Repo(repo_dir)
        url = repo.remotes.origin.url
        if url not in self.data:
            local_repo = {
                'commit': repo.heads[0].commit.hexsha,
                'workingDir': repo_dir.split('/')[-1],
            }
            collection = getJsonFromPmr(url)
            if len(info:=collection.get('items', {})) > 0:
                local_repo = {
                    **local_repo,
                    **{
                        'id': info[0]['data'][0]['value'],
                        'title': info[0]['data'][1]['value'],
                        'owner': info[0]['data'][2]['value'],
                        'description': info[0]['data'][3]['value'],
                        'storage': info[0]['data'][4]['value'],
                        'version': collection['version'],
                        'subModels': [{'name': sm.name, 'workspace': sm.url, 'commit': sm.hexsha} for sm in repo.submodules]
                    }
                }
            self.data[url] = local_repo

    # synchronising workspace in local
    def __synchroWorkspace(self, url):
        workingDir = self.data[url]['workingDir']
        path = os.path.join(WORKSPACE_DIR, workingDir)
        repo = git.Repo(path)
        repoCommit = repo.heads[0].commit.hexsha
        try:
            remoteCommit = self.__getRemoteCommit(url)
            if remoteCommit != repoCommit:
                # pull or update local workspace
                g = git.cmd.Git(path)
                g.pull()
                self.data[url]['commit'] = remoteCommit
        except Exception as e:
            print('\n', e, '\n\t', url, ', repo commit: %s' %
                  (repoCommit), ', workingDir: ', workingDir, '\n')

    # clone workspaces based on provided URL
    def __cloneWorkspace(self, url):
        path = os.path.join(CURRENT_PATH, WORKSPACE_DIR)
        lisNumericFolder = [int(name)
                            for name in os.listdir(path) if name.isnumeric()]
        workingDir = str(max(lisNumericFolder) +
                         1) if len(lisNumericFolder) > 0 else '1'
        path = os.path.join(path, workingDir)
        try:
            repo = git.Repo.clone_from(url, path)
            self.__local_repo_info(path, repo)
        except git.exc.GitError as e:
            print('\n', e, '\n')
        except:
            print("Unknown clone error: ", url)

    def __getRemoteCommit(self, url):
        g = git.cmd.Git()
        ref = g.ls_remote(url).split('\n')
        hashRef = ref[0].split('\t')
        return hashRef[0]

    def __get_exposures(self):
        for url in tqdm(self.getListExposures(fromServer=True)):
            exposure = getJsonFromPmr(url)
            for link in exposure['links']:
                if link['prompt'] == 'Workspace URL':
                    wks_url = link['href']
                    if wks_url in self.data:
                        for d in exposure['items'][0]['data']:
                            if d['name'] == 'commit_id':
                                if 'exposures' not in self.data[wks_url]:
                                    self.data[wks_url]['exposures'] = {}
                                self.data[wks_url]['exposures'][url] = d['value']
                                break
                    else:
                        logging.error(f'Cannot find workspace {wks_url} for exposure {url} ')
                    break

    def __updateRdf(self):
        graph = rdflib.Graph()
        rdfPaths = getAllFilesInDir(WORKSPACE_DIR)
        for rdfPath in rdfPaths:
            if rdfPath.endswith('.rdf'):
                try:
                    graph.parse(rdfPath, format='application/rdf+xml')
                except:
                    print('error', rdfPath)
        dumpPickle(graph, CURRENT_PATH, RESOURCE_PATH, 'rdf.graph')

    def addCellml(self, id=None, url=None, cellmlId=None):
        if id is not None:
            url = self.getUrl(id)
        if url in self.data:
            if 'cellml' in self.data[url]:
                if cellmlId not in self.data[url]['cellml']:
                    self.data[url]['cellml'] += [cellmlId]
            else:
                self.data[url]['cellml'] = [cellmlId]

    def extract(self, sysCellmls):
        for _, data in sysCellmls.getData().items():
            self.addCellml(url=data['workspace'], cellmlId=data['id'])
        self.dumpJson()

