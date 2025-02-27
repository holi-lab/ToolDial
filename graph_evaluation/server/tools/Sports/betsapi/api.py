import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bet365_inplay_filter(league_id: int=None, sport_id: int=1, skip_esports: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "bet365 inplay filter"
    
    """
    url = f"https://betsapi2.p.rapidapi.com/v1/bet365/inplay_filter"
    querystring = {}
    if league_id:
        querystring['league_id'] = league_id
    if sport_id:
        querystring['sport_id'] = sport_id
    if skip_esports:
        querystring['skip_esports'] = skip_esports
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bet365_inplay(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "bet365 inplay data"
    
    """
    url = f"https://betsapi2.p.rapidapi.com/v1/bet365/inplay"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bet365_inplay_event(fi: str, stats: str=None, lineup: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "inplay event with all scores/stats/markets"
    
    """
    url = f"https://betsapi2.p.rapidapi.com/v1/bet365/event"
    querystring = {'FI': fi, }
    if stats:
        querystring['stats'] = stats
    if lineup:
        querystring['lineup'] = lineup
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bet365_upcoming_events(sport_id: int, day: int=None, league_id: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get bet365 fixtures"
    
    """
    url = f"https://betsapi2.p.rapidapi.com/v1/bet365/upcoming"
    querystring = {'sport_id': sport_id, }
    if day:
        querystring['day'] = day
    if league_id:
        querystring['league_id'] = league_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bet365_prematch_odds(fi: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "prematch odds"
    
    """
    url = f"https://betsapi2.p.rapidapi.com/v3/bet365/prematch"
    querystring = {'FI': fi, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bet365_result(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "to view bet365 event result"
    
    """
    url = f"https://betsapi2.p.rapidapi.com/v1/bet365/result"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsapi2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

