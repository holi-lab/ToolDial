import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_event_details(is_id: str, start: int=2020, end: int=2025, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details for an event"
    id: The event ID.
        
    """
    url = f"https://checkiday.p.rapidapi.com/event/{is_id}"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checkiday.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_event_s(query: str, adult: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for events."
    query: The search query
        adult: Include content that may be unsafe for children or for viewing at work
        
    """
    url = f"https://checkiday.p.rapidapi.com/search"
    querystring = {'query': query, }
    if adult:
        querystring['adult'] = adult
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checkiday.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_today_s_events(timezone: str='America/Chicago', adult: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today's events"
    timezone: IANA timezone
        adult: Include content that may be unsafe for children or for viewing at work
        
    """
    url = f"https://checkiday.p.rapidapi.com/today"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if adult:
        querystring['adult'] = adult
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checkiday.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_events_for_a_date(date: str, timezone: str='America/Chicago', adult: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lists of events happening today, multi-day events starting today, and multi-day events that are still happening today."
    date: The date to fetch events for
        timezone: IANA timezone
        adult: Include content that may be unsafe for children or for viewing at work
        
    """
    url = f"https://checkiday.p.rapidapi.com/events"
    querystring = {'date': date, }
    if timezone:
        querystring['timezone'] = timezone
    if adult:
        querystring['adult'] = adult
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checkiday.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

