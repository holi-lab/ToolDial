import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def futures_odds(event_id: int, state: str='nj', date_format: str='iso', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve a list of futures odds for a given event."
    event_id: Event to include. Get a list of event IDs from futures/events endpoint.
        state: Set the state to show sportsbooks solely from. Will show nationwide sportsbooks if omitted.
        date_format: Set date format to UNIX Timestamp or ISO. Defaults to UNIX if omitted.
        
    """
    url = f"https://oddsboom.p.rapidapi.com/futures/odds/"
    querystring = {'event_id': event_id, }
    if state:
        querystring['state'] = state
    if date_format:
        querystring['date_format'] = date_format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oddsboom.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def futures_events(league: str, date_format: str='iso', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve a list of futures events."
    league: League to include, such as NBA, NFL etc.
        date_format: Set date format to UNIX Timestamp or ISO. Defaults to UNIX if omitted.
        
    """
    url = f"https://oddsboom.p.rapidapi.com/futures/events/"
    querystring = {'league': league, }
    if date_format:
        querystring['date_format'] = date_format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oddsboom.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(event_id: int, state: str='nj', markets: str='money,spread,total', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve a list of odds for a given game."
    event_id: The Event ID to show. Get event IDs from the events endpoint.
        state: Set the state to show sportsbooks solely from. Will show nationwide sportsbooks if omitted.
        markets: Markets to show. For example, money for moneyline. If not set, all markets will be shown.

Value can be playerprops to only show player props if sport allows. This option can only be used on its own.
        
    """
    url = f"https://oddsboom.p.rapidapi.com/odds/"
    querystring = {'event_id': event_id, }
    if state:
        querystring['state'] = state
    if markets:
        querystring['markets'] = markets
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oddsboom.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(league: str, date_format: str='unix', limit: int=2, sport: str='basketball', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve a list of games, in date order."
    league: League to include, such as NBA, NFL etc.
        date_format: Set date format to UNIX Timestamp or ISO. Defaults to UNIX if omitted.
        limit: Limit the number of events shown.
        
    """
    url = f"https://oddsboom.p.rapidapi.com/events/"
    querystring = {'league': league, }
    if date_format:
        querystring['date_format'] = date_format
    if limit:
        querystring['limit'] = limit
    if sport:
        querystring['sport'] = sport
    if league:
        querystring['league'] = league
    if date_format:
        querystring['date_format'] = date_format
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oddsboom.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve all available market types (for example money line, spread, first-quarter total)."
    sport: Sport to include, such as basketball, football.
        
    """
    url = f"https://oddsboom.p.rapidapi.com/markets/"
    querystring = {'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oddsboom.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

