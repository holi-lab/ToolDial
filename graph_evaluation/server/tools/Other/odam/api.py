import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def info(dataset: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information on the dataset
		"
    dataset: dataset identifier
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/infos/{dataset}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tsv6(dataset: str, category: str, subset: str, format: str='tsv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the variable list within the specified category of a merged data subsets along with the metadata
		"
    dataset: dataset identifier
        category: category
        subset: data subset identifier
        format: output format
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/query/{dataset}/({subset})/{category}"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tsv4(subset: str, dataset: str, format: str='tsv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the entry list  of a merged data subsets
		"
    subset: data subset identifier
        dataset: dataset identifier
        format: output format
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/query/{dataset}/({subset})/entry"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def build(dataset: str, debug: int=0, format: str='tsv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Building a database and indexing it, allows to accelerate considerably the response times on large files
		"
    dataset: dataset identifier
        debug: debug status
        format: output format
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/build/{dataset}"
    querystring = {}
    if debug:
        querystring['debug'] = debug
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tsv3(dataset: str, subset: str, debug: int=0, format: str='tsv', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all values of a merged data subsets
		"
    dataset: dataset identifier
        subset: data subset identifier
        debug: debug status
        format: output format
        limit: limit the number of lines in the output table (0 means no limit)
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/query/{dataset}/({subset})"
    querystring = {}
    if debug:
        querystring['debug'] = debug
    if format:
        querystring['format'] = format
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tsv5(value: str, subset: str, entry: str, dataset: str, limit: int=10, format: str='tsv', debug: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all values of a merged data subsets
		"
    value: value as filter criteria for entry attribute
        subset: data subset identifier
        entry: entry identifier
        dataset: dataset identifier
        limit: limit the number of lines in the output table (0 means no limit)
        format: output format
        debug: debug status
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/query/{dataset}/({subset})/{entry}/{value}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if format:
        querystring['format'] = format
    if debug:
        querystring['debug'] = debug
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tsv(dataset: str, format: str='tsv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the subset list of a dataset along with the metadata
		"
    dataset: dataset identifier
        format: output format
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/query/{dataset}"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check(dataset: str, format: str='tsv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Made automatically many test checks and return a list several status about the dataset implementation
		"
    dataset: dataset identifier
        format: output format
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/check/{dataset}"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tsv2(subset: str, dataset: str, limit: int=10, debug: int=0, format: str='tsv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all values of a data subset
		"
    subset: data subset identifier
        dataset: dataset identifier
        limit: limit the number of lines in the output table (0 means no limit)
        debug: debug status
        format: output format
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/query/{dataset}/{subset}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if debug:
        querystring['debug'] = debug
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tsv1(dataset: str, format: str='tsv', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all attribute metadata for all subsets of a dataset
		"
    dataset: dataset identifier
        format: output format
        
    """
    url = f"https://odam.p.rapidapi.com/getdata/query/{dataset}/metadata"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

