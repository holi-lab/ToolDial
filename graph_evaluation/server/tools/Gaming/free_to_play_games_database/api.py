import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def filter_games_by_multiple_tags(tag: str, platform: str='pc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Filter Games by multiple tags and platform for personalized results. Optionally you can also use the "platform" and "sort" parameters."
    
    """
    url = f"https://free-to-play-games-database.p.rapidapi.com/api/filter"
    querystring = {'tag': tag, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-to-play-games-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_by_platform(platform: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert platform, eg: pc, browser."
    
    """
    url = f"https://free-to-play-games-database.p.rapidapi.com/api/games?"
    querystring = {'platform': platform, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-to-play-games-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_by_category_or_tag(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert game category or tag, eg: mmorpg, shooter, pvp, mmofps and more."
    
    """
    url = f"https://free-to-play-games-database.p.rapidapi.com/api/games?"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-to-play-games-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_by_platform_category_sorted(category: str='mmorpg', platform: str='browser', sort_by: str='release-date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get games list using multiple parameters"
    
    """
    url = f"https://free-to-play-games-database.p.rapidapi.com/api/games?"
    querystring = {}
    if category:
        querystring['category'] = category
    if platform:
        querystring['platform'] = platform
    if sort_by:
        querystring['sort-by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-to-play-games-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def return_details_from_a_specific_game(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert game id"
    
    """
    url = f"https://free-to-play-games-database.p.rapidapi.com/api/game"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-to-play-games-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sort_games(sort_by: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert sort by, eg: release-date, alphabetical or relevance"
    
    """
    url = f"https://free-to-play-games-database.p.rapidapi.com/api/games?"
    querystring = {'sort-by': sort_by, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-to-play-games-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all games!"
    
    """
    url = f"https://free-to-play-games-database.p.rapidapi.com/api/games"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-to-play-games-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

