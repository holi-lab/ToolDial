import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_specific_team(page: int=1, q: str='Golden State Warriors', limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the specific team data"
    page: The current page, used for navigating thru the list
        q: The team name. Can be a keyword. i.e. Golden (will return all teams with `golden` in their name)
        limit: The maximum of items to return per page, a max of 100
        
    """
    url = f"https://nbi-data.p.rapidapi.com/team/search"
    querystring = {}
    if page:
        querystring['page'] = page
    if q:
        querystring['q'] = q
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nbi-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams(page: int=1, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the teams"
    page: The current page, used for navigating thru the list
        limit: The maximum of items to return per page, a max of 100
        
    """
    url = f"https://nbi-data.p.rapidapi.com/team"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nbi-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_player(q: str='Stephen Curry', page: int=1, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player data"
    q: The player name to search. OPTIONAL. Leave blank to get all players.
        page: The current page, used for navigating thru the list.
        limit: The maximum of items to return per page, a maximum of 100
        
    """
    url = f"https://nbi-data.p.rapidapi.com/player"
    querystring = {}
    if q:
        querystring['q'] = q
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nbi-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

