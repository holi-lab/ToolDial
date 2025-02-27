import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_1_inplay_events(league_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "soccer inplay events"
    
    """
    url = f"https://betsapi-soccer.p.rapidapi.com/v1/events/inplay"
    querystring = {}
    if league_id:
        querystring['league_id'] = league_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_upcoming_events(skip_esports: bool=None, league_id: int=None, team_id: int=None, cc: str=None, page: int=None, day: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "upcoming soccer events"
    
    """
    url = f"https://betsapi-soccer.p.rapidapi.com/v2/events/upcoming"
    querystring = {}
    if skip_esports:
        querystring['skip_esports'] = skip_esports
    if league_id:
        querystring['league_id'] = league_id
    if team_id:
        querystring['team_id'] = team_id
    if cc:
        querystring['cc'] = cc
    if page:
        querystring['page'] = page
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_ended_events(skip_esports: bool=None, league_id: int=None, cc: str=None, page: int=None, team_id: int=None, day: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ended soccer events"
    
    """
    url = f"https://betsapi-soccer.p.rapidapi.com/v2/events/ended"
    querystring = {}
    if skip_esports:
        querystring['skip_esports'] = skip_esports
    if league_id:
        querystring['league_id'] = league_id
    if cc:
        querystring['cc'] = cc
    if page:
        querystring['page'] = page
    if team_id:
        querystring['team_id'] = team_id
    if day:
        querystring['day'] = day
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_4_event_view(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "event details for upcoming/inplay/ended event"
    
    """
    url = f"https://betsapi-soccer.p.rapidapi.com/v1/event/view"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_league(page: int=None, cc: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "soccer league list"
    
    """
    url = f"https://betsapi-soccer.p.rapidapi.com/v1/league"
    querystring = {}
    if page:
        querystring['page'] = page
    if cc:
        querystring['cc'] = cc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_team(page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Soccer team list"
    
    """
    url = f"https://betsapi-soccer.p.rapidapi.com/v1/team"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-soccer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

