import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_patents_authors(authors_id: int, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get existing patents for a given Author based on Author Id"
    
    """
    url = f"https://patenteye.p.rapidapi.com/authors/patents"
    querystring = {'authors_id': authors_id, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ipc(ipc_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get IPC information for a specific IPC class based on IPC code"
    
    """
    url = f"https://patenteye.p.rapidapi.com/ipc/"
    querystring = {'ipc_code': ipc_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_organizations(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Organizations based on the presence of keywords in the Organization Name"
    
    """
    url = f"https://patenteye.p.rapidapi.com/organizations/search"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def semantic_search_abstract(query: str, n_examples: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use query based semantic search of claims to retrieve interesting patents. The response will contain the Patent Id and the respective Abstract for the top n_examples most relevant results."
    
    """
    url = f"https://patenteye.p.rapidapi.com/patents/abstract/semanticsearch"
    querystring = {'query': query, }
    if n_examples:
        querystring['n_examples'] = n_examples
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_ipc(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for an existing IPC code based on the presence of specific keywords in the Name"
    
    """
    url = f"https://patenteye.p.rapidapi.com/ipc/search"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_topics_over_time(start_date: str='2022-01-01', topic_id: int=None, end_date: str='2022-03-30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the frequency of existing topics over a time window."
    
    """
    url = f"https://patenteye.p.rapidapi.com/topics/time"
    querystring = {}
    if start_date:
        querystring['start_date'] = start_date
    if topic_id:
        querystring['topic_id'] = topic_id
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topic_search_keywords(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search existing topics based on the presence of keywords"
    
    """
    url = f"https://patenteye.p.rapidapi.com/topics/search"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_topics(is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get existing topics or specific topic information based on Topic Id. The response contains a list of topics with the respective frequency and keywords."
    
    """
    url = f"https://patenteye.p.rapidapi.com/topics/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_topics_subclass(subclass_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get existing topics per IPC Subclass based on Subclass Name"
    
    """
    url = f"https://patenteye.p.rapidapi.com/topics/subclass"
    querystring = {'subclass_name': subclass_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_topics_class(class_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get existing topics per IPC Class based on Class Name"
    
    """
    url = f"https://patenteye.p.rapidapi.com/topics/class"
    querystring = {'class_name': class_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_topics_organization(organization_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get existing topics for an Organization based on Organization Id"
    
    """
    url = f"https://patenteye.p.rapidapi.com/topics/organization"
    querystring = {'organization_id': organization_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def semantic_search_claims(query: str, n_examples: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use query based semantic search of claims to retrieve interesting patents. The response will contain the Patent Id and the respective Claims for the top n_examples most relevant results."
    
    """
    url = f"https://patenteye.p.rapidapi.com/patents/claims/semanticsearch"
    querystring = {'query': query, }
    if n_examples:
        querystring['n_examples'] = n_examples
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_patent_abstract(patent_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Abstract for a patent based on Patent Id"
    
    """
    url = f"https://patenteye.p.rapidapi.com/patents/abstract"
    querystring = {'patent_id': patent_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_patent_claims(patent_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get patent Claims based on Patent Id"
    
    """
    url = f"https://patenteye.p.rapidapi.com/patents/claims"
    querystring = {'patent_id': patent_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_topics_section(section_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get existing topics per IPC Section based on Section Name"
    
    """
    url = f"https://patenteye.p.rapidapi.com/topics/section"
    querystring = {'section_name': section_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keyword_search_abstract(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for patents based on the presence of keywords in the Abstract."
    
    """
    url = f"https://patenteye.p.rapidapi.com/patents/abstract/search"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keyword_search_title(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for patents based on the presence of keywords in the Title."
    
    """
    url = f"https://patenteye.p.rapidapi.com/patents/search"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_patents(topic_id: int=None, patent_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get patent(s) information based on Patent Id or Topic Id"
    
    """
    url = f"https://patenteye.p.rapidapi.com/patents/"
    querystring = {}
    if topic_id:
        querystring['topic_id'] = topic_id
    if patent_id:
        querystring['patent_id'] = patent_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keyword_search_claims(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for patents based on the presence of keywords in the Claims."
    
    """
    url = f"https://patenteye.p.rapidapi.com/patents/claims/search"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_patents_organization(organization_id: int, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get existing patents for a given Organization based on Organization Id"
    
    """
    url = f"https://patenteye.p.rapidapi.com/organizations/patents"
    querystring = {'organization_id': organization_id, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_organizations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all existing Organizations. Any Organization with <100 patents is re-labeled as "Individual"."
    
    """
    url = f"https://patenteye.p.rapidapi.com/organizations/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_authors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all existing Authors"
    
    """
    url = f"https://patenteye.p.rapidapi.com/authors/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_authors(names: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Authors based on the presence of keywords in the Author Name"
    
    """
    url = f"https://patenteye.p.rapidapi.com/authors/search"
    querystring = {'names': names, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "patenteye.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

