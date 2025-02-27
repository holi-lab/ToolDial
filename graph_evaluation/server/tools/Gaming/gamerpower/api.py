import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def filter_group_giveaways(platform: str, type: str='game.loot', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Filter and group platforms and giveaway types to get personalized results."
    
    """
    url = f"https://gamerpower.p.rapidapi.com/api/filter"
    querystring = {'platform': platform, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamerpower.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_giveaway_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details from a specific giveaway (insert giveaway id)."
    
    """
    url = f"https://gamerpower.p.rapidapi.com/api/giveaway"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamerpower.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_giveaways_by_platform(platform: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert platform eg: pc, steam, epic-games-store, uplay, gog, icthio, ps4, xbox-one, switch, android, ios, vr, battlenet"
    
    """
    url = f"https://gamerpower.p.rapidapi.com/api/giveaways?"
    querystring = {'platform': platform, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamerpower.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_giveaways_by_type(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live giveaways by type, eg: game, loot, beta"
    
    """
    url = f"https://gamerpower.p.rapidapi.com/api/giveaways?"
    querystring = {'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamerpower.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sort_live_giveaways(sort_by: str='value', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert sort by, eg: date, value, popularity"
    
    """
    url = f"https://gamerpower.p.rapidapi.com/api/giveaways?"
    querystring = {}
    if sort_by:
        querystring['sort-by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamerpower.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_giveaways_by_platform_type_sorted(platform: str='steam', sort_by: str='popularity', type: str='loot', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all live giveaways by platform and type and sorted."
    
    """
    url = f"https://gamerpower.p.rapidapi.com/api/giveaways?"
    querystring = {}
    if platform:
        querystring['platform'] = platform
    if sort_by:
        querystring['sort-by'] = sort_by
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamerpower.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def total_live_giveaways_worth_estimation_in_dollars(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can also use the "platform" and "type" parameters to get more specific results, eg: /api/worth?platform=steam?type=game"
    
    """
    url = f"https://gamerpower.p.rapidapi.com/api/worth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamerpower.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_live_giveaways(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all live giveaways."
    
    """
    url = f"https://gamerpower.p.rapidapi.com/api/giveaways"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gamerpower.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

