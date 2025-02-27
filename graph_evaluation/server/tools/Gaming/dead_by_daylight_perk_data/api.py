import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def perk(perk: str, clean: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a perk by name."
    
    """
    url = f"https://dead-by-daylight-perk-data.p.rapidapi.com/perk/{perk}/{clean}"
    querystring = {}
    if clean:
        querystring['clean'] = clean
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dead-by-daylight-perk-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def perk_search(searchterm: str, clean: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to search for perks based on a search term. If you want to find all perks that apply expose, this is the place to do it."
    
    """
    url = f"https://dead-by-daylight-perk-data.p.rapidapi.com/perkSearch/{searchterm}/{clean}"
    querystring = {}
    if clean:
        querystring['clean'] = clean
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dead-by-daylight-perk-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def owner(owner: str, clean: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets perks by perk owner."
    
    """
    url = f"https://dead-by-daylight-perk-data.p.rapidapi.com/owner/{owner}/{clean}"
    querystring = {}
    if clean:
        querystring['clean'] = clean
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dead-by-daylight-perk-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

