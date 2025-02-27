import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def export(year: int, resolution: str, hs_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get export-data by year or month"
    
    """
    url = f"https://indonesia-import-export.p.rapidapi.com/export-data"
    querystring = {'year': year, 'resolution': resolution, 'hs_code': hs_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-import-export.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def import(year: int, resolution: str, hs_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get import-data by year or month"
    
    """
    url = f"https://indonesia-import-export.p.rapidapi.com/import-data"
    querystring = {'year': year, 'resolution': resolution, 'hs_code': hs_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-import-export.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hs_detail(hs_code: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "hs-detail"
    
    """
    url = f"https://indonesia-import-export.p.rapidapi.com/hs-detail"
    querystring = {'hs_code': hs_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-import-export.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hs_search(offsite: str, limit: str, keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search hs code"
    
    """
    url = f"https://indonesia-import-export.p.rapidapi.com/hs-search"
    querystring = {'offsite': offsite, 'limit': limit, 'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indonesia-import-export.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

