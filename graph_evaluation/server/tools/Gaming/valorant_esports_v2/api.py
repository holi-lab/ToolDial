import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_player_info(playerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get name, country, team and more about a player."
    
    """
    url = f"https://valorant-esports1.p.rapidapi.com/v1/players/{playerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "valorant-esports1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_players(minrounds: str=None, timespan: str=None, map: str=None, event: str=None, agent: str=None, minrating: str=None, region: str=None, country: str=None, event_series: str=None, limit: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get id, name and country of all players."
    
    """
    url = f"https://valorant-esports1.p.rapidapi.com/v1/players"
    querystring = {}
    if minrounds:
        querystring['minrounds'] = minrounds
    if timespan:
        querystring['timespan'] = timespan
    if map:
        querystring['map'] = map
    if event:
        querystring['event'] = event
    if agent:
        querystring['agent'] = agent
    if minrating:
        querystring['minrating'] = minrating
    if region:
        querystring['region'] = region
    if country:
        querystring['country'] = country
    if event_series:
        querystring['event_series'] = event_series
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "valorant-esports1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_info(teamid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comprehensive information about a team, including player details, events, results, and more."
    teamid: Team id from vlr.gg
        
    """
    url = f"https://valorant-esports1.p.rapidapi.com/v1/teams/{teamid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "valorant-esports1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

