import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_region_detail(slug: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get region detail with different language."
    slug: You can get this values get regions endpoint method.
        
    """
    url = f"https://league-of-legends-champions.p.rapidapi.com/regions/{lang}/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_statics(period: str, tier: str='platinum', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you champion statics. This statics can filtered time period and league."
    tier: You can filtered league name for example ;
Iron, Bronze, Silver, Gold, Platinum, Diamond, Master, Grandmaster, Challenger
        
    """
    url = f"https://league-of-legends-champions.p.rapidapi.com/statics"
    querystring = {'period': period, }
    if tier:
        querystring['tier'] = tier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_regions(lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get all league of legends regions with different languages"
    
    """
    url = f"https://league-of-legends-champions.p.rapidapi.com/regions/{lang}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champion_detail(name: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get champion all information with different language"
    
    """
    url = f"https://league-of-legends-champions.p.rapidapi.com/champions/{lang}/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champions(size: int, page: int, lang: str, role: str='fighter', name: str='aatrox', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can list of league of legends champions with different language.This api have a pagination you can get max 10 champions for per request."
    size: you can get max 10 champions per request
        
    """
    url = f"https://league-of-legends-champions.p.rapidapi.com/champions/{lang}"
    querystring = {'size': size, 'page': page, }
    if role:
        querystring['role'] = role
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

