import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def raw_data_download(key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download the specified raw data CSV. The `key` is the raw data CSV file name provided by the Raw data - List API.
		Note that this endpoint will redirect to a S3 URL for download, and should not be used within the browser due to CORS restriction."
    
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/download-raw-data"
    querystring = {'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, platform: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Chrome extensions by name or ID"
    
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/search"
    querystring = {'q': q, }
    if platform:
        querystring['platform'] = platform
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ranking_extensions(namespace: str, platform: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a list of extensions, sorted by their ranking in the given namespace.
		
		A namespace can be the overall-rank, or the type of item in Chrome Web Store (extension, app, or theme), or a category."
    namespace: Should be one of the namespaces in /api/metastats
        page: Page number (starting from 0)
        
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/ranking"
    querystring = {'namespace': namespace, }
    if platform:
        querystring['platform'] = platform
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_versions(platform: str=None, is_id: str='bmnlcjabgnpnenekpadlanbbkooimhnj', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of downloadable versions for an extension"
    
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/list-versions"
    querystring = {}
    if platform:
        querystring['platform'] = platform
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download(is_id: str, version: str, type: str, platform: str='chrome,edge,firefox', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download an extension"
    type: Either ZIP or CRX
        
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/download"
    querystring = {'id': is_id, 'version': version, 'type': type, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def raw_data_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the available raw data CSVs for downloads"
    
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/raw-data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ranking_publishers(namespace: str, platform: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of publishers in a given namespace. Namespace should be one of the fields in Metastats - Publishers API response."
    namespace: Should be the inform of \"authors-*<key>*-rank\" where *<key>* is obtained from the /api/authors-metastats API response
        page: Page number (starting from 0)
        
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/authors-ranking"
    querystring = {'namespace': namespace, }
    if platform:
        querystring['platform'] = platform
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews(is_id: str, platform: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve user reviews"
    
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/reviews"
    querystring = {'id': is_id, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trends(is_id: str, platform: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical data trends"
    
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/trends"
    querystring = {'id': is_id, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detail(is_id: str, platform: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details of a Chrome extension"
    
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/detail"
    querystring = {'id': is_id, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manifest(is_id: str, platform: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Chrome manifest file"
    
    """
    url = f"https://chrome-stats.p.rapidapi.com/api/manifest"
    querystring = {'id': is_id, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chrome-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

