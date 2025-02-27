import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def names_base_on_gender(gender: int, namelength: int=2, count: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Names base on gender"
    namelength: Length of Chinese name, 1 means random length between 2 or 3, specify length is 2 or 3
        count: Number of results returned, range from 1 to 5
        
    """
    url = f"https://chinese-names.p.rapidapi.com/names/gender"
    querystring = {'gender': gender, }
    if namelength:
        querystring['nameLength'] = namelength
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chinese-names.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def common_names(namelength: int=2, count: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get common names"
    namelength: Length of Chinese name, 1 means random length between 2 or 3, specify length is 2 or 3
        count: Number of results returned, range from 1 to 5
        
    """
    url = f"https://chinese-names.p.rapidapi.com/names/common"
    querystring = {}
    if namelength:
        querystring['nameLength'] = namelength
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chinese-names.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def names_base_on_season(season: str, count: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Names base on season"
    count: Number of results returned, range from 1 to 5
        
    """
    url = f"https://chinese-names.p.rapidapi.com/names/season"
    querystring = {'season': season, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chinese-names.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nickname(count: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "nickname"
    count: Number of results returned, range from 1 to 5
        
    """
    url = f"https://chinese-names.p.rapidapi.com/names/nickname"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chinese-names.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

