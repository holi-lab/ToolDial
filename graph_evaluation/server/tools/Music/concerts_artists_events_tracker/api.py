import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def past_events_of_a_venue_concert(maxdate: str, mindate: str, name: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access past events by name of venue/concert and time period."
    
    """
    url = f"https://concerts-artists-events-tracker.p.rapidapi.com/venue/past"
    querystring = {'maxDate': maxdate, 'minDate': mindate, 'name': name, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concerts-artists-events-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_location(maxdate: str, mindate: str, name: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access upcoming events by a location name and time period."
    
    """
    url = f"https://concerts-artists-events-tracker.p.rapidapi.com/location"
    querystring = {'maxDate': maxdate, 'minDate': mindate, 'name': name, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concerts-artists-events-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def past_events_of_artist(mindate: str, maxdate: str, name: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access past events and concerts by artist name and period"
    
    """
    url = f"https://concerts-artists-events-tracker.p.rapidapi.com/artist/past"
    querystring = {'minDate': mindate, 'maxDate': maxdate, 'name': name, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concerts-artists-events-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upcoming_events_by_venue_concert_name(name: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access upcoming event by the name of the venue/concert"
    
    """
    url = f"https://concerts-artists-events-tracker.p.rapidapi.com/venue"
    querystring = {'name': name, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concerts-artists-events-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upcoming_events_of_artist(name: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search through upcoming events of artists by their name."
    
    """
    url = f"https://concerts-artists-events-tracker.p.rapidapi.com/artist"
    querystring = {'name': name, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "concerts-artists-events-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

