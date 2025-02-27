import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcharacterbyname(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get specific Character by name"
    
    """
    url = f"https://marvel-heroic-api-unlock-the-mcu-legendary-characters.p.rapidapi.com/name"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marvel-heroic-api-unlock-the-mcu-legendary-characters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcharacterbyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sepcific character"
    
    """
    url = f"https://marvel-heroic-api-unlock-the-mcu-legendary-characters.p.rapidapi.com/characters/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marvel-heroic-api-unlock-the-mcu-legendary-characters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def characters(pageno: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All 30 Characters"
    
    """
    url = f"https://marvel-heroic-api-unlock-the-mcu-legendary-characters.p.rapidapi.com/characters"
    querystring = {}
    if pageno:
        querystring['pageNo'] = pageno
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "marvel-heroic-api-unlock-the-mcu-legendary-characters.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

