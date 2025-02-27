import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def some_operation_od(secret: str, match_id: int, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting match statistics"
    secret: Your API Secret that you get from your account on our website
        match_id: Match ID
        key: Your API Key that you get from your account on our website API key
        
    """
    url = f"https://test2527.p.rapidapi.com/matches/stats.json"
    querystring = {'secret': secret, 'match_id': match_id, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2527.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def some_operation_od(secret: str, key: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting match events"
    secret: Your API Secret that you get from your account on our website
        key: Your API Key that you get from your account on our website API key
        id: Match ID
        
    """
    url = f"https://test2527.p.rapidapi.com/scores/events.json"
    querystring = {'secret': secret, 'key': key, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2527.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def some_operation_od(secret: str, key: str, competition_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Goalscorers list"
    secret: Your API Secret that you get from your account on our website
        key: Your API Key that you get from your account on our website API key
        competition_id: Competition ID
        
    """
    url = f"https://test2527.p.rapidapi.com/api-client/competitions/goalscorers.json"
    querystring = {'secret': secret, 'key': key, }
    if competition_id:
        querystring['competition_id'] = competition_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2527.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def some_operation_od(secret: str, competition_id: int, key: str, lang: str=None, season: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Competitions standings"
    secret: Your API Secret that you get from your account on our website
        competition_id: Filter by competition
        key: Your API Key that you get from your account on our website API key
        lang: Translate into language
        season: Season with standings
        
    """
    url = f"https://test2527.p.rapidapi.com/leagues/table.json"
    querystring = {'secret': secret, 'competition_id': competition_id, 'key': key, }
    if lang:
        querystring['lang'] = lang
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2527.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def some_operation_od(secret: str, match_id: int, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting match lineups"
    secret: Your API Secret that you get from your account on our website
        match_id: Match ID
        key: Your API Key that you get from your account on our website API key
        
    """
    url = f"https://test2527.p.rapidapi.com/matches/lineups.json"
    querystring = {'secret': secret, 'match_id': match_id, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2527.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def some_operation_od(secret: str, key: str, team2_id: int, team1_id: int, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "H2H team comparison"
    secret: Your API Secret that you get from your account on our website
        key: Your API Key that you get from your account on our website API key
        team2_id: Team 2
        team1_id: Team 1
        lang: Translate into language
        
    """
    url = f"https://test2527.p.rapidapi.com/teams/head2head.json"
    querystring = {'secret': secret, 'key': key, 'team2_id': team2_id, 'team1_id': team1_id, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2527.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

