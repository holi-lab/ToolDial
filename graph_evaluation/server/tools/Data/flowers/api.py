import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def show_image(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show image of the flower"
    
    """
    url = f"https://flowers2.p.rapidapi.com/{is_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flowers2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list(pageindex: str='0', pagesize: str='10', name: str='African Daisy', scientificname: str='Osteospermum', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all flowers"
    pagesize: Boundary is up to 500.
        name: Search by containing name
        scientificname: Search by containing scientific name
        
    """
    url = f"https://flowers2.p.rapidapi.com/list"
    querystring = {}
    if pageindex:
        querystring['pageIndex'] = pageindex
    if pagesize:
        querystring['pageSize'] = pagesize
    if name:
        querystring['name'] = name
    if scientificname:
        querystring['scientificName'] = scientificname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flowers2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all info about the flower"
    
    """
    url = f"https://flowers2.p.rapidapi.com/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flowers2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

