import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def comparison_of_urls(url1: str, url2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Semantically compares the specified sources."
    url1: The URL/Network path to the document to parse.
        url2: The URL/Network path to the document to parse.
        
    """
    url = f"https://intellexer.p.rapidapi.com/compareUrls"
    querystring = {'url1': url1, 'url2': url2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "intellexer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summarization_of_urls(url: str, summaryrestriction: int, structure: str='General', conceptsrestriction: int=10, fulltexttrees: bool=None, loadconceptstree: bool=None, loadnamedentitytree: bool=None, returnedtopicscount: int=3, usepercentrestriction: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return summary data for a document from a given URL."
    url: URL for processing.
        summaryrestriction: Determine size of a summary measured in sentences.
        structure: Specify structure of the document (News Article, Research Paper, Patent or General).
        conceptsrestriction: Determine the length of a concept tree.
        fulltexttrees: Load full text trees.
        loadconceptstree: Load a tree of concepts (FALSE by default).
        loadnamedentitytree: Load a tree of Named Entities (FALSE by default).
        returnedtopicscount: Determine max count of document topics to return.
        usepercentrestriction: Use percentage of the number of sentences in the original text instead of the exact number of sentences.
        
    """
    url = f"https://intellexer.p.rapidapi.com/summarize?url={url}&loadConceptsTree={loadconceptstree}&loadNamedEntityTree={loadnamedentitytree}&summaryRestriction={summaryrestriction}&usePercentRestriction={usepercentrestriction}&conceptsRestriction={conceptsrestriction}&structure={structure}&returnedTopicsCount={returnedtopicscount}&fullTextTrees={fulltexttrees}"
    querystring = {}
    if structure:
        querystring['structure'] = structure
    if conceptsrestriction:
        querystring['conceptsrestriction'] = conceptsrestriction
    if fulltexttrees:
        querystring['fulltexttrees'] = fulltexttrees
    if loadconceptstree:
        querystring['loadconceptstree'] = loadconceptstree
    if loadnamedentitytree:
        querystring['loadnamedentitytree'] = loadnamedentitytree
    if returnedtopicscount:
        querystring['returnedtopicscount'] = returnedtopicscount
    if usepercentrestriction:
        querystring['usepercentrestriction'] = usepercentrestriction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "intellexer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def concept_extraction_from_urls(url: str, conceptsrestriction: int, fulltexttrees: bool=None, loadsentences: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return tree of concepts for a document from a given URL."
    url: URL for process.
        conceptsrestriction: Determine the length of a concept tree.
        fulltexttrees: Load full text trees.
        loadsentences: Load all sentences.
        
    """
    url = f"https://intellexer.p.rapidapi.com/clusterize"
    querystring = {'url': url, 'conceptsRestriction': conceptsrestriction, }
    if fulltexttrees:
        querystring['fullTextTrees'] = fulltexttrees
    if loadsentences:
        querystring['loadSentences'] = loadsentences
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "intellexer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def named_entity_recognition_from_urls(url: str, loadnamedentities: bool=None, loadsentences: bool=None, loadrelationstree: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Load named entities (personal names, names of organizations, geographical locations, positions/occupations, etc.) from a document from a given URL."
    url: Load text from given url.
        loadnamedentities: Load named entities (FALSE by default).
        loadsentences: Load source sentences (FALSE by default).
        loadrelationstree: Load tree of relations (FALSE by default).
        
    """
    url = f"https://intellexer.p.rapidapi.com/recognizeNe"
    querystring = {'url': url, }
    if loadnamedentities:
        querystring['loadNamedEntities'] = loadnamedentities
    if loadsentences:
        querystring['loadSentences'] = loadsentences
    if loadrelationstree:
        querystring['loadrelationstree'] = loadrelationstree
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "intellexer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text_extraction_from_urls(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parse URL and extract content in the form of plain text."
    
    """
    url = f"https://intellexer.p.rapidapi.com/parse?url={url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "intellexer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

