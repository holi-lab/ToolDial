import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def teams_history_teamids(teamids: str='145', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query baseball team history by ID"
    teamids: Enter a team ID - can be found in the game's endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/teams-history"
    querystring = {}
    if teamids:
        querystring['teamIds'] = teamids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def free_agents_seasonid(seasonid: str='2012', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query baseball free  agents"
    seasonid: Enter a season year
        
    """
    url = f"https://baseball4.p.rapidapi.com/freeagents"
    querystring = {}
    if seasonid:
        querystring['seasonId'] = seasonid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_playerids(personids: str='676265', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Baseball players"
    personids: Enter a player ID
        
    """
    url = f"https://baseball4.p.rapidapi.com/players"
    querystring = {}
    if personids:
        querystring['personIds'] = personids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_personnel_teamids(teamids: str='145', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query baseball team history by ID"
    teamids: Enter a team ID - can be found in the game's endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/teams-personnel"
    querystring = {}
    if teamids:
        querystring['teamIds'] = teamids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_roster_teamids(teamids: str='145', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query baseball team roster by ID"
    teamids: Enter a team ID - can be found in the game's endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/teams-roster"
    querystring = {}
    if teamids:
        querystring['teamIds'] = teamids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule_date(date: str='2021-07-30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Baseball schedule"
    date: Enter a date: YYYY-MM-DD
        
    """
    url = f"https://baseball4.p.rapidapi.com/schedule-date"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game(gamepk: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query Baseball games, teams, scores etc..."
    gamepk: Enter a game ID - can be found in the schedule endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/game"
    querystring = {'gamePk': gamepk, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_playbyplay_gamepk(gamepk: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query Baseball games, teams, scores etc..."
    gamepk: Enter a game ID - can be found in the schedule endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/game-playbyplay"
    querystring = {'gamePk': gamepk, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_matrix_gamepk(gamepk: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query Baseball scores, stats, betting information etc..."
    gamepk: Enter a game ID - can be found in the schedule endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/game-matrix"
    querystring = {'gamePk': gamepk, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_probability_gamepk(gamepk: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query Baseball probability data."
    gamepk: Enter a game ID - can be found in the schedule endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/game-probability"
    querystring = {'gamePk': gamepk, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_boxscore_gamepk(gamepk: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query Baseball games, teams, scores etc..."
    gamepk: Enter a game ID - can be found in the schedule endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/game-boxscore"
    querystring = {'gamePk': gamepk, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Baseball schedule"
    
    """
    url = f"https://baseball4.p.rapidapi.com/schedule"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_affiliates_teamids(teamids: str='145', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query baseball team affiliates by ID"
    teamids: Enter a team ID - can be found in the game's endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/teams-affiliates"
    querystring = {}
    if teamids:
        querystring['teamIds'] = teamids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_coaches_teamids(teamids: str='145', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query baseball team coaches by ID"
    teamids: Enter a team ID - can be found in the game's endpoint
        
    """
    url = f"https://baseball4.p.rapidapi.com/teams-coaches"
    querystring = {}
    if teamids:
        querystring['teamIds'] = teamids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Baseball leagues"
    
    """
    url = f"https://baseball4.p.rapidapi.com/leagues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Baseball venues"
    
    """
    url = f"https://baseball4.p.rapidapi.com/venues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_seasonid(seasonid: str='2021', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Baseball seasons"
    seasonid: Enter a season year, ex: 2010
        
    """
    url = f"https://baseball4.p.rapidapi.com/seasons"
    querystring = {}
    if seasonid:
        querystring['seasonId'] = seasonid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baseball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

