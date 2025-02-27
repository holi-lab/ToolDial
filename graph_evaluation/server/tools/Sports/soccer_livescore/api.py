import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_leagues(league: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all league name in day"
    league: Input league from Get League API
        
    """
    url = f"https://soccer-livescore.p.rapidapi.com/v1/global/getleague"
    querystring = {}
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-livescore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def finished_matches(league: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information of finished matches"
    league: Input league from Get League API
        
    """
    url = f"https://soccer-livescore.p.rapidapi.com/v1/global/getsoccerscorefinish"
    querystring = {}
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-livescore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pending_matches(league: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information of pending matches"
    league: Input league from Get League API
        
    """
    url = f"https://soccer-livescore.p.rapidapi.com/v1/global/getsoccerscorepending"
    querystring = {}
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-livescore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_matches(league: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information of living matches"
    league: Input league from Get League API
        
    """
    url = f"https://soccer-livescore.p.rapidapi.com/v1/global/getsoccerscorelive"
    querystring = {}
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccer-livescore.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

