import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gameweekgamesbydatetime(datetime: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides Gameweek games from a date time in the following format '2023-09-16T14:00'"
    
    """
    url = f"https://premier-league-stats.p.rapidapi.com/gameWeekGamesByDateTime"
    querystring = {'dateTime': datetime, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gameweekgamesbyseason(gameweek: str, season: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all games from a particular gameweek and season."
    
    """
    url = f"https://premier-league-stats.p.rapidapi.com/gameWeekGamesBySeason"
    querystring = {'gameweek': gameweek, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gameweekgamesbyid(gameweek: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get games from a particular gameweek by id."
    
    """
    url = f"https://premier-league-stats.p.rapidapi.com/gameWeekGamesById"
    querystring = {'gameweek': gameweek, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gameweekgamesbyclubandseason(club: str, season: str, gameweek: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all games from a particular gameweek, club and season."
    
    """
    url = f"https://premier-league-stats.p.rapidapi.com/gameWeekGamesByClubAndSeason"
    querystring = {'club': club, 'season': season, 'gameweek': gameweek, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballers_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a players stats by their ID. Choose from all 597 registered premier league players."
    
    """
    url = f"https://premier-league-stats.p.rapidapi.com/footballersById"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballersbyclub(club: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of footballers for a given club."
    
    """
    url = f"https://premier-league-stats.p.rapidapi.com/footballersByClub"
    querystring = {'club': club, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def footballersbyname(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of footballers with given name."
    
    """
    url = f"https://premier-league-stats.p.rapidapi.com/footballersByName"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

