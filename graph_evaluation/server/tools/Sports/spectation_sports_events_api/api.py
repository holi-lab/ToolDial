import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fighter(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single fighter by ID"
    
    """
    url = f"https://spectation-sports-events-api.p.rapidapi.com/fighters/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spectation-sports-events-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fighters(weightdivision: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of fighters"
    
    """
    url = f"https://spectation-sports-events-api.p.rapidapi.com/fighters"
    querystring = {}
    if weightdivision:
        querystring['weightDivision'] = weightdivision
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spectation-sports-events-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fight(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single fight by ID"
    
    """
    url = f"https://spectation-sports-events-api.p.rapidapi.com/fights/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spectation-sports-events-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fights(outcome: str=None, sport: str=None, type: str='upcoming', category: str=None, event: int=None, page: int=1, fighter: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch fights for upcoming/previous events"
    category: Category/Weight division of the fights
        event: The ID of the event.
        fighter: The ID of the fighter.
        
    """
    url = f"https://spectation-sports-events-api.p.rapidapi.com/fights"
    querystring = {}
    if outcome:
        querystring['outcome'] = outcome
    if sport:
        querystring['sport'] = sport
    if type:
        querystring['type'] = type
    if category:
        querystring['category'] = category
    if event:
        querystring['event'] = event
    if page:
        querystring['page'] = page
    if fighter:
        querystring['fighter'] = fighter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spectation-sports-events-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single event by ID"
    
    """
    url = f"https://spectation-sports-events-api.p.rapidapi.com/events/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spectation-sports-events-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(enclosuretype: str=None, type: str='previous', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch upcoming/previous events"
    enclosuretype: Type of enclosure. Available variables:
ring
cage
        type: Shows upcoming events only by default
        
    """
    url = f"https://spectation-sports-events-api.p.rapidapi.com/events"
    querystring = {}
    if enclosuretype:
        querystring['enclosureType'] = enclosuretype
    if type:
        querystring['type'] = type
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spectation-sports-events-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

