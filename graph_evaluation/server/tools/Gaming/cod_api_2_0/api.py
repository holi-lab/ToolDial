import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def season_skill_rating_mw_2_ranked(season: str, limit: str='250', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season Skill Rating for top 250 players, MW 2 ranked Leaderboard"
    season: Example: s1
        limit: Provide a number from 1 to 250
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/ranked/mp/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_career_leaderboard_wz2(season: str, limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season career information for up to 2000 players on the leaderboard."
    season: Example: s1
        limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/career/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_kills_leaderboard_wz2(season: str, limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season kills information for up to 2000 players on the leaderboard."
    season: Example: s1
        limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/kills/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_wins_leaderboard_wz2(season: str, limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season wins information for up to 2000 players on the leaderboard."
    season: Example: s1
        limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/placement/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def career_leaderboard_wz2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top career information for up to 2000 players on the leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/career"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_critical_damage_leaderboard_wz2(season: str, limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season critical damage information for up to 2000 players on the leaderboard."
    season: Example: s1
        limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/critical/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def critical_damage_leaderboard_wz2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top critical damage information for up to 2000 players on the leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/critical"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_critical_damage_leaderboard_mw2(season: str, limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season critical damage information for up to 2000 players on the MW 2 leaderboard."
    season: Example: s1
        limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/mode/critical/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def critical_damage_leaderboard_mw2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top critical damage information for up to 2000 players on the MW 2 leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/mode/critical"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wins_leaderboard_wz2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top wins information for up to 2000 players on the leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/placement"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_kills_leaderboard_mw_2(season: str, limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season kills information for up to 2000 players on the MW 2 leaderboard."
    season: Example: s1
        limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/mode/kills/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kills_leaderboard_mw2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top kills information for up to 2000 players on the MW 2 leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/mode/kills"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def skill_rating_mw_2_ranked(limit: str='250', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top Skill Rating for top 250 players, MW 2 ranked Leaderboard"
    limit: Provide a number from 1 to 250
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/ranked/mp"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_career_leaderboard_mw_2(limit: int=2000, season: str='s1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season career information for up to 2000 players on the MW 2 leaderboard."
    limit: Provide a number of players from 1 to 2000
        season: Example: s1
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/mode/career/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_wins_leaderboard_mw_2(limit: int, season: str='s1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season wins information for up to 2000 players on the MW 2 leaderboard."
    limit: Provide a number of players from 1 to 2000
        season: Example: s1
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/mode/wins/season/{season}"
    querystring = {'limit': limit, }
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def career_leaderboard_mw2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top career information for up to 2000 players on the MW 2 leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/mode/career"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wins_leaderboard_mw2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top wins information for up to 2000 players on the MW 2 leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/mw2/mode/wins"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_gulag_leaderboard_wz2(season: str, limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season gulag information for up to 2000 players on the leaderboard."
    season: Example: s1
        limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/gulag/season/{season}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gulag_leaderboard_wz2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top gulag information for up to 2000 players on the leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/gulag"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kills_leaderboard_wz2(limit: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top kills information for up to 2000 players on the leaderboard."
    limit: Provide a number of players from 1 to 2000
        
    """
    url = f"https://cod-api-2-0.p.rapidapi.com/wz2/mode/kills"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cod-api-2-0.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

