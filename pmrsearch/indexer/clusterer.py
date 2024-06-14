from lxml import etree
import os
import json
from bs4 import BeautifulSoup
import codecs
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import numpy as np
from collections import OrderedDict

FEAT_DOCUMENTATION = 0
FEAT_XPATH_DEEP = 1
FEAT_XPATH_WIDE = 2
FEAT_XPATH_STRUCT = 3
FEAT_XPATH_DEEP_WIDE_STRUCT = 4
FEAT_ALL = 5

class CellmlClusterer:
    """
    A class to cluster cellml documents using HDBSCAN.
    Input arguments:
     - cellmls : a dictionary consisting data contained by RS_CELLML
     - featureType : the type of feature being use for clustering
       (the possible value: FEAT_DOCUMENTATION, FEAT_XPATH_DEEP, FEAT_XPATH_WIDE,
        FEAT_XPATH_STRUCT, FEAT_XPATH_DEEP_WIDE_STRUCT, FEAT_ALL)
    """
    def __init__(self, workspace_dir, cellmls, featureType=FEAT_XPATH_DEEP_WIDE_STRUCT):
        self.__workspace_dir = workspace_dir
        self.featureType = featureType
        self.__clusteringCellmls(cellmls)

    def __clusteringCellmls(self, cellmls):
        documentations = self.__getCellmlsDocumentation(cellmls)
        self.cellmlUrls, cellmlDocs = zip(*documentations.items())

        self.tfidf = self.__calcuateTfidf(cellmlDocs)
        self.pairwiseSimilarity = self.tfidf * self.tfidf.T
        n, _ = self.pairwiseSimilarity.shape
        self.pairwiseSimilarity[np.arange(n), np.arange(n)] = -1.0

        self.__runClustering()

    def __getCellmlsDocumentation(self, cellmls):
        documentations = {}
        parser = etree.XMLParser(recover=True, remove_comments=True)
        for key, cellml in cellmls.items():
            if key not in documentations:
                documentations[key] = ''
            path = os.path.join(self.__workspace_dir,cellml['workingDir'],cellml['cellml'])
            root = etree.parse(path, parser).getroot()
            tree = etree.ElementTree(root)

            # delete all rdf
            for rdf in tree.xpath("//*[local-name()='RDF']"):
                rdf.getparent().remove(rdf)

            if self.featureType == FEAT_DOCUMENTATION:
                documentations[key] += self.__clusterDocumentation(root, path)
            elif self.featureType == FEAT_XPATH_DEEP:
                documentations[key] += self.__clusterXPathDeep(root, tree)
            elif self.featureType == FEAT_XPATH_WIDE:
                documentations[key] += self.__clusterXPathWide(root)
            elif self.featureType == FEAT_XPATH_STRUCT:
                documentations[key] += self.__clusterXPathStruct(root)
            elif self.featureType == FEAT_XPATH_DEEP_WIDE_STRUCT:
                documentations[key] += self.__clusterXPathDeep(root, tree)
                documentations[key] += self.__clusterXPathWide(root)
                documentations[key] += self.__clusterXPathStruct(root)
            elif self.featureType == FEAT_ALL:
                documentations[key] += self.__clusterDocumentation(root, path)
                documentations[key] += self.__clusterXPathDeep(root, tree)
                documentations[key] += self.__clusterXPathWide(root)
                documentations[key] += self.__clusterXPathStruct(root)
        return documentations

    def __clusterDocumentation(self, root, path):
        def checkSeparateDocs(path):
            dirName = os.path.dirname(os.path.realpath(path))
            documentation = []
            for fileName in os.listdir(dirName):
                if fileName.endswith('.html') or fileName.endswith('.md'):
                    htmlPath = os.path.join(dirName,fileName)
                    try:
                        file = codecs.open(htmlPath, "r", "utf-8")
                        soup = BeautifulSoup(file.read(), 'lxml')
                    except:
                        file = codecs.open(htmlPath, "r", "latin-1")
                        soup = BeautifulSoup(file.read(), "lxml")
                    splitText = soup.text.split('\n')
                    for line in splitText:
                        if line.strip() not in documentation:
                            documentation += [line.strip()]
            return '. '.join(documentation)


        documents = root.xpath("//*[local-name()='documentation']")
        documentation = ''
        if len(documents) > 0:
            reslist = list(documents[0].iter())
            documentation += '. '.join([element.text for element in reslist if isinstance(element.text, str) and len(element.text.strip())>0])
        else:
            documentation += ' ' + checkSeparateDocs(path)
        return documentation

    def __clusterXPathDeep(self, root, tree):
        """The extracted features are the longest XPath (DEEP)"""
        documentation = ''
        for e in root.iter():
            if len(e.getchildren()) == 0:
                try:
                    if 'documentation' not in tree.getelementpath(e):
                        xPath = re.sub('\[[0-9]+\]','',tree.getelementpath(e))
                        xPath = re.sub('\{(.*?)}','',xPath)
                        xPath = xPath.replace('/','_')
                        documentation += ' ' + xPath

                except:
                    pass
        return documentation

    def __clusterXPathWide(self, root):
        """The extracted features is each element at level 1 such as unit,
        component, connection, import, and group widely"""
        documentation = ''
        for e in root:
            try:
                if e.tag.endswith(('units','connection','import','group')):
                    xPath = e.tag
                    for child in e:
                        xPath += '_' + child.tag
                    xPath = xPath.replace('/','_')
                    documentation += ' ' + xPath
                if e.tag.endswith(('component')):
                    xPath = e.tag
                    for child in e:
                        xPath += '_' + child.tag
                        if child.tag.endswith(('math')):
                            for nChild in child:
                                xPath += '_' + nChild.tag
                    documentation += ' ' + xPath
            except:
                pass
        return re.sub('\{(.*?)}','',documentation)

    def __clusterXPathStruct(self, root):
        """The extracted the overal structure of cellml started from level 1"""
        documentation = ''
        #clear xml, get structure only
        for e in root.iter():
            e.text = ''
            for k in e.attrib:
                del e.attrib[k]

        for e in root:
            try:
                if not e.tag.endswith(('documentation','RDF')):
                    xPath = re.sub('>\s+<', '><', etree.tostring(e).decode('ascii').replace('\n',''))
                    xPath = re.sub('\s+(.*?)>', '>', xPath)
                    documentation += ' ' + xPath
            except:
                pass
        return re.sub('\{(.*?)}','',documentation)

    def __calcuateTfidf(self, cellmlDocs):
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        return vect.fit_transform(cellmlDocs)

    def __runClustering(self):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True).fit(self.tfidf.A)

        self.clusteredDocs = {}
        self.url2Cluster = {}
        for i in range(len(self.clusterer.labels_)):
            if self.clusterer.labels_[i] not in self.clusteredDocs:
                self.clusteredDocs[self.clusterer.labels_[i]] = []
            self.clusteredDocs[self.clusterer.labels_[i]] += [self.cellmlUrls[i]]
            self.url2Cluster[self.cellmlUrls[i]] = self.clusterer.labels_[i]

    def getSimCellmlsByCluster(self, cellmlUrl, minSim=0.7):
        results = {}
        if cellmlUrl in self.url2Cluster:
            clusterNum = self.url2Cluster[cellmlUrl]
            if clusterNum > -1:
                idxCellml = self.cellmlUrls.index(cellmlUrl)
                for cellmlUrlPair in self.clusteredDocs[clusterNum]:
                    idxCellmlPair = self.cellmlUrls.index(cellmlUrlPair)
                    if self.pairwiseSimilarity[idxCellml,idxCellmlPair] >= minSim:
                        results[cellmlUrlPair] = self.pairwiseSimilarity[idxCellml,idxCellmlPair]
            else:
                return {}
        return OrderedDict({k: v for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)})

    def getSimCellmlsByVector(self, cellmlUrl, minSim=0.7):
        results = OrderedDict()
        if cellmlUrl in self.cellmlUrls:
            idxCellml = self.cellmlUrls.index(cellmlUrl)
            valSimCellmls = self.pairwiseSimilarity[idxCellml].toarray()[0]
            idxSimCellmls = valSimCellmls.argsort()[:][::-1]
            for idxSimCellml in idxSimCellmls:
                if valSimCellmls[idxSimCellml] < minSim:
                    break
                results[self.cellmlUrls[idxSimCellml]] = valSimCellmls[idxSimCellml]
        return results

    def getCellmlClusterType(self, cellmlUrl):
        if cellmlUrl in self.url2Cluster:
            return self.url2Cluster[cellmlUrl]
        return None

    def predictCellmlsCluster(self, cellmls=None):
        documentations = self.__getCellmlsDocumentation(cellmls)
        cellmlUrls, cellmlDocs = zip(*documentations.items())
        tfidf = self.__calcuateTfidf(cellmlDocs)
        test_labels, strengths = hdbscan.approximate_predict(self.clusterer, tfidf.A)
        return test_labels

    def getDict(self):
        clusteredDocs = {str(k):v for k,v in self.clusteredDocs.items()}
        url2Cluster = {k:str(v) for k,v in self.url2Cluster.items()}
        return {'cluster':clusteredDocs, 'url2Cluster':url2Cluster}
