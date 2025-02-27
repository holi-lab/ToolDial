import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_healthcheck_health_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check health status of components of the API (database...)"
    
    """
    url = f"https://ecoindex.p.rapidapi.com/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoindex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_host_list_version_hosts_get(version: str, q: str=None, date_from: str=None, size: int=50, date_to: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns a list of hosts that ran an ecoindex analysis order by most request made"
    version: Engine version used to run the analysis (v0 or v1)
        q: Filter by partial host name
        date_from: Start date of the filter elements (example: 2020-01-01)
        size: Number of elements per page
        date_to: End date of the filter elements  (example: 2020-01-01)
        page: Page number
        
    """
    url = f"https://ecoindex.p.rapidapi.com/{version}/hosts"
    querystring = {}
    if q:
        querystring['q'] = q
    if date_from:
        querystring['date_from'] = date_from
    if size:
        querystring['size'] = size
    if date_to:
        querystring['date_to'] = date_to
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoindex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_screenshot_version_ecoindexes_id_screenshot_get(version: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns the screenshot of the webpage analysis if it exists"
    version: Engine version used to run the analysis (v0 or v1)
        
    """
    url = f"https://ecoindex.p.rapidapi.com/{version}/ecoindexes/{is_id}/screenshot"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoindex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ecoindex_analysis_list_version_ecoindexes_get(version: str, size: int=50, host: str=None, date_from: str=None, page: int=1, date_to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns a list of ecoindex analysis corresponding to query filters and the given version engine. The results are ordered by ascending date"
    version: Engine version used to run the analysis (v0 or v1)
        size: Number of elements per page
        host: Host name you want to filter
        date_from: Start date of the filter elements (example: 2020-01-01)
        page: Page number
        date_to: End date of the filter elements  (example: 2020-01-01)
        
    """
    url = f"https://ecoindex.p.rapidapi.com/{version}/ecoindexes"
    querystring = {}
    if size:
        querystring['size'] = size
    if host:
        querystring['host'] = host
    if date_from:
        querystring['date_from'] = date_from
    if page:
        querystring['page'] = page
    if date_to:
        querystring['date_to'] = date_to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoindex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ecoindex_analysis_by_id_version_ecoindexes_id_get(version: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns an ecoindex given by its unique identifier"
    version: Engine version used to run the analysis (v0 or v1)
        
    """
    url = f"https://ecoindex.p.rapidapi.com/{version}/ecoindexes/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoindex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ecoindex_analysis_task_by_id_v1_tasks_ecoindexes_id_get(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns an ecoindex given by its unique identifier"
    
    """
    url = f"https://ecoindex.p.rapidapi.com/v1/tasks/ecoindexes/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecoindex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

