import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_player_details(name: str, region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Player Details based on UserName.
		Case Sensitive!"
    
    """
    url = f"https://league-of-legends-galore.p.rapidapi.com/api/getPlayerRank"
    querystring = {'name': name, 'region': region, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-galore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_champion(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will go through the list of champions and return a single RANDOM champion!"
    
    """
    url = f"https://league-of-legends-galore.p.rapidapi.com/api/randomChamp"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-galore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_champion_tier_list(rank: str='master', region: str='kr', tier: str='s+', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get either a list of the current champion tier list or a tier list section based on YOUR request!"
    
    """
    url = f"https://league-of-legends-galore.p.rapidapi.com/api/getChampTierList"
    querystring = {}
    if rank:
        querystring['rank'] = rank
    if region:
        querystring['region'] = region
    if tier:
        querystring['tier'] = tier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-galore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_league_ranks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get League of Legends ranks and their player base!"
    
    """
    url = f"https://league-of-legends-galore.p.rapidapi.com/api/getLoLRanks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-galore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_a_champion(releasedate: str=None, name: str='zed', resource: str=None, rp: str=None, blueessence: str=None, attribute: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will go through the list of champions and return a single champion or a list of champions depending on YOUR request!"
    
    """
    url = f"https://league-of-legends-galore.p.rapidapi.com/api/selectChamp"
    querystring = {}
    if releasedate:
        querystring['releaseDate'] = releasedate
    if name:
        querystring['name'] = name
    if resource:
        querystring['resource'] = resource
    if rp:
        querystring['rp'] = rp
    if blueessence:
        querystring['blueEssence'] = blueessence
    if attribute:
        querystring['attribute'] = attribute
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-galore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_item(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will go through the list of items and return a single RANDOM item!"
    
    """
    url = f"https://league-of-legends-galore.p.rapidapi.com/api/randomItem"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-galore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_a_item(champsynergy: str=None, itemsynergy: str=None, price: str=None, name: str='sword', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will go through the list of items and return a single item or a list of items depending on YOUR request!"
    
    """
    url = f"https://league-of-legends-galore.p.rapidapi.com/api/selectItem"
    querystring = {}
    if champsynergy:
        querystring['champSynergy'] = champsynergy
    if itemsynergy:
        querystring['itemSynergy'] = itemsynergy
    if price:
        querystring['price'] = price
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-galore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

