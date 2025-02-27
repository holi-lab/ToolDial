import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_tournaments(name: str=None, is_id: str=None, location: str=None, tier: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint tournaments allows you to get the next tournaments."
    
    """
    url = f"https://dota-24.p.rapidapi.com/tournaments"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if location:
        querystring['location'] = location
    if tier:
        querystring['tier'] = tier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dota-24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team(is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/teams?id=id` endpoint allows you to consume the information of a team according to its id."
    
    """
    url = f"https://dota-24.p.rapidapi.com/teams"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dota-24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams(name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint teams allows you to consume all the teams in active or inactive state."
    
    """
    url = f"https://dota-24.p.rapidapi.com/teams"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dota-24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_matches(firstteam: str=None, startdate: str=None, secondteam: str=None, tournament: str=None, status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint matches allows you to get live and upcoming games."
    
    """
    url = f"https://dota-24.p.rapidapi.com/matches"
    querystring = {}
    if firstteam:
        querystring['firstTeam'] = firstteam
    if startdate:
        querystring['startDate'] = startdate
    if secondteam:
        querystring['secondTeam'] = secondteam
    if tournament:
        querystring['tournament'] = tournament
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dota-24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_items(is_id: str=None, price: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint items allows you to obtain the game items."
    
    """
    url = f"https://dota-24.p.rapidapi.com/items"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if price:
        querystring['price'] = price
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dota-24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_heroes(name: str=None, is_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint heroes allows to obtain a list with all the heroes of the game."
    
    """
    url = f"https://dota-24.p.rapidapi.com/heroes"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dota-24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

