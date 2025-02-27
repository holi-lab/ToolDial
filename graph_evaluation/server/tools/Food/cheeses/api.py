import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list(pagesize: str='10', pageindex: str='0', name: str='Mozzarella', exactregionname: str='Savoie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all cheeses"
    pagesize: Boundary is up to 500.
        
    """
    url = f"https://cheeses.p.rapidapi.com/list"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if pageindex:
        querystring['pageIndex'] = pageindex
    if name:
        querystring['name'] = name
    if exactregionname:
        querystring['exactRegionName'] = exactregionname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cheeses.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all info about the cheese"
    
    """
    url = f"https://cheeses.p.rapidapi.com/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cheeses.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

