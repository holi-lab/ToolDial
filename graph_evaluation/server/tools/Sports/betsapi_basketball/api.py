import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_2_upcoming_events(team_id: int=None, cc: str=None, page: int=None, day: int=None, league_id: int=None, skip_esports: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "upcoming soccer events"
    
    """
    url = f"https://betsapi-basketball.p.rapidapi.com/v2/events/upcoming"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if cc:
        querystring['cc'] = cc
    if page:
        querystring['page'] = page
    if day:
        querystring['day'] = day
    if league_id:
        querystring['league_id'] = league_id
    if skip_esports:
        querystring['skip_esports'] = skip_esports
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_ended_events(page: int=None, league_id: int=None, cc: str=None, day: int=None, team_id: int=None, skip_esports: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ended basketball events"
    
    """
    url = f"https://betsapi-basketball.p.rapidapi.com/v2/events/ended"
    querystring = {}
    if page:
        querystring['page'] = page
    if league_id:
        querystring['league_id'] = league_id
    if cc:
        querystring['cc'] = cc
    if day:
        querystring['day'] = day
    if team_id:
        querystring['team_id'] = team_id
    if skip_esports:
        querystring['skip_esports'] = skip_esports
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-basketball.p.rapidapi.com"
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
    url = f"https://betsapi-basketball.p.rapidapi.com/v1/event/view"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_inplay_events(league_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "basketball inplay events"
    
    """
    url = f"https://betsapi-basketball.p.rapidapi.com/v1/events/inplay"
    querystring = {}
    if league_id:
        querystring['league_id'] = league_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

