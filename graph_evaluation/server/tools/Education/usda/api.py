import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def topics_single_topic(api_key: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a single topic node."
    api_key: api key
        id: An integer parameter that specifies the ID of the node returned. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Topics"
    querystring = {'api_key': api_key, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topics_child_node_count(api_key: str, is_id: int, getchildren: bool, getcount: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a total count of all child topic nodes."
    api_key: api key
        id: An integer parameter that specifies the ID of the partent topic node for which child nodes are returned. (FromUri)
        getchildren: A boolean parameter that specifies to return child nodes. When [true], child nodes are returned. When [false], the parent specified parent node is returned. (FromUri)
        getcount: A boolean parameter that indicates to return a count. Specifying a false value will return zero. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Topics"
    querystring = {'api_key': api_key, 'id': is_id, 'getChildren': getchildren, 'getCount': getcount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amberwaves_single_node_content(api_key: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a single amber waves content node."
    api_key: api key
        id: An integer parameter that specifies the ID of the node returned. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/AmberWaves"
    querystring = {'api_key': api_key, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_retrieve_chart_aliases(api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a collection of valid chart collection aliases."
    api_key: api key
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Charts"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topics_node_count(api_key: str, getcount: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a total count of all topic nodes."
    api_key: api key
        getcount: getCount: A boolean parameter that indicates to return a count. Specifying a false value will return zero. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Topics"
    querystring = {'api_key': api_key, 'getCount': getcount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amberwaves_section_aliases(api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all amber waves section aliases."
    api_key: api key
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Amber Waves"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amberwaves_total_count(api_key: str, getcount: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a total count of all amber waves content nodes."
    api_key: api key
        getcount: A boolean parameter that indicates to return a count. Specifying a false value will return zero. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Amber Waves"
    querystring = {'api_key': api_key, 'getCount': getcount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_first_100_nodes(api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a collection of the first 100 chart content nodes, sorted by descending release date."
    api_key: api key
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Charts"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topics_content_node_collection(api_key: str, size: int, start: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a paged collection of topic content nodes."
    api_key: api key
        size: An integer parameter that specifies the desired page size. (FromUri)
        start: A zero-index integer parameter that specifies the desired starting index (0 for page 1, 10 for page 2, etc...) (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Topics"
    querystring = {'api_key': api_key, 'size': size, 'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chart_retrieve_content(api_key: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a single chart content node."
    api_key: api key
        id: An integer parameter that specifies the ID of the node returned. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Charts"
    querystring = {'api_key': api_key, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amberwaves_collection_of_content_nodes(api_key: str, size: int, start: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a paged collection of amber waves content nodes, sorted by descending release date."
    api_key: api key
        size: An integer parameter that specifies the desired page size. (FromUri)
        start: A zero-index integer parameter that specifies the desired starting index (0 for page 1, 100 for page 2, etc...) (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Amber Waves"
    querystring = {'api_key': api_key, 'size': size, 'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amberwaves_retrieve_paged_collection(alias: str, start: int, api_key: str, size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a paged collection of amber waves content nodes filtered by a particular section alias, ordered by descending release date."
    alias: A string parameter that specifies a series alias used for filtering. (FromUri)
        start: A zero-index integer parameter that specifies the desired starting index (0 for page 1, 100 for page 2, etc...) (FromUri)
        api_key: api key
        size: An integer parameter that specifies the desired page size. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Amber Waves"
    querystring = {'alias': alias, 'start': start, 'api_key': api_key, 'size': size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_total_count(api_key: str, getcount: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a total count of all chart content nodes."
    api_key: api key
        getcount: A boolean parameter that indicates to return a count. Specifying a false value will return zero. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Charts"
    querystring = {'api_key': api_key, 'getCount': getcount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amberwaves_first_100_nodes(api_key: str, alias: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the first 100 amber waves nodes filtered by a particular section alias, ordered by descending release date."
    api_key: api key
        alias: A string parameter that specifies a series alias used for filtering. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Amber Waves"
    querystring = {'api_key': api_key, 'alias': alias, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_retrieve_nodes(api_key: str, size: int, start: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a paged collection of chart content nodes, sorted by descending release date."
    api_key: api key
        size: An integer parameter that specifies the desired page size. (FromUri)
        start: A zero-index integer parameter that specifies the desired starting index (0 for page 1, 10 for page 2, etc...) (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Charts"
    querystring = {'api_key': api_key, 'size': size, 'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amberwaves_first_100_content_nodes(api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the first 100 amber waves content nodes, sorted by descending release date."
    api_key: api key
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/AmberWaves"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amberwaves_filtered_count(api_key: str, alias: str, getcount: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a total count of amber waves content nodes filtered by a particular section alias."
    api_key: api key
        alias: A string parameter that specifies a series alias used for filtering. (FromUri)
        getcount: A boolean parameter that indicates to return a count. Specifying a false value will return zero. (FromUri)
        
    """
    url = f"https://t14ha70d-usda-v1.p.rapidapi.com/content/Amber Waves"
    querystring = {'api_key': api_key, 'alias': alias, 'getCount': getcount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t14ha70d-usda-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

