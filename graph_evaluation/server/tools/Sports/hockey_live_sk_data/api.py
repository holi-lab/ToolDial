import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def game_report(is_id: int, tz: str='America/New_York', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get game score, goals, penalties and team statistics for a certain game ID"
    id: ID of the game from the game´s schedule
        tz: Show the times and dates in the provided timezone
        
    """
    url = f"https://hockey-live-sk-data.p.rapidapi.com/game/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-live-sk-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_game_report_by_teams(home_team: str, league: str, away_team: str, tz: str='America/New_York', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the last game report of the two teams"
    home_team: 3-char acronym of the home team
        away_team: 3-char acronym of the away team
        tz: Show the times and dates in the provided timezone
        
    """
    url = f"https://hockey-live-sk-data.p.rapidapi.com/game/{home_team}/{away_team}/{league}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-live-sk-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_schedule(league: str, year: int, tz: str='America/New_York', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get games list for certain league and year"
    year: Tournament year
        tz: Show the times and dates in the provided timezone
        
    """
    url = f"https://hockey-live-sk-data.p.rapidapi.com/games/{league}/{year}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-live-sk-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_statistics(name: str, league: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player historical statistics by his name or his ID"
    name: Name of the player or his ID
        
    """
    url = f"https://hockey-live-sk-data.p.rapidapi.com/player/{name}/{league}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-live-sk-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_standings(league: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current or historical team standings for the certain league and year"
    year: Tournament year
        
    """
    url = f"https://hockey-live-sk-data.p.rapidapi.com/table/{league}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-live-sk-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

