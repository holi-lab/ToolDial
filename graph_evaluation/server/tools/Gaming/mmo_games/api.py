import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def return_details_from_a_specific_game(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert game id"
    
    """
    url = f"https://mmo-games.p.rapidapi.com/game"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_mmo_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest MMO News powered by MMOBomb."
    
    """
    url = f"https://mmo-games.p.rapidapi.com/latestnews"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_mmo_giveaways(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live Giveaways list"
    
    """
    url = f"https://mmo-games.p.rapidapi.com/giveaways"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter_games_by_multiple_tags(tag: str, platform: str='pc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Filter Games by multiple tags for personalized results. Insert tag, eg: mmorpg, shooter, pvp, mmofps and more at [https://www.mmobomb.com/api ](url). Optionally you can also use the "platform" and "sort" parameters"
    
    """
    url = f"https://mmo-games.p.rapidapi.com/filter?"
    querystring = {'tag': tag, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_by_platform_category_sorted(sort_by: str='release-date', platform: str='browser', category: str='mmorpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Games by platform & category & sorted."
    
    """
    url = f"https://mmo-games.p.rapidapi.com/games?"
    querystring = {}
    if sort_by:
        querystring['sort-by'] = sort_by
    if platform:
        querystring['platform'] = platform
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sort_games(sort_by: str='alphabetical', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sort games by release date, alphabetical, popularity or relevance"
    
    """
    url = f"https://mmo-games.p.rapidapi.com/games?"
    querystring = {}
    if sort_by:
        querystring['sort-by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_by_category_or_tag(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert game category or tag, eg: mmorpg, shooter, pvp, mmofps and more. Full tag list at [https://www.mmobomb.com/api](url)"
    
    """
    url = f"https://mmo-games.p.rapidapi.com/games?"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_by_platform(platform: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert platform, eg: pc, browser or all"
    
    """
    url = f"https://mmo-games.p.rapidapi.com/games?"
    querystring = {'platform': platform, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live games list."
    
    """
    url = f"https://mmo-games.p.rapidapi.com/games"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mmo-games.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

