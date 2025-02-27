import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def artist_search(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for artists by name using full text search. Sorted by relevancy."
    
    """
    url = f"https://community-songkick.p.rapidapi.com/search/artists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-songkick.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist_s_upcoming_events_calendar(artist_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find upcoming events for an artist. Use our artist search to find an artist ID."
    
    """
    url = f"https://community-songkick.p.rapidapi.com/artists/{artist_id}/calendar.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-songkick.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venue_search(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for venues on Songkick using full text search, including past names and aliases. Sorted by relevancy."
    
    """
    url = f"https://community-songkick.p.rapidapi.com/search/venues.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-songkick.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venue_s_upcoming_events_calendar(venue_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find upcoming events for a venue. Use our venue search to find a venue ID."
    
    """
    url = f"https://community-songkick.p.rapidapi.com/venues/{venue_id}/calendar.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-songkick.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_upcoming_events_calendar(username: str, reason: str='tracked_artist', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find upcoming events on a user’s calendar. There is another API request for past events."
    
    """
    url = f"https://community-songkick.p.rapidapi.com/users/{username}/calendar.json"
    querystring = {}
    if reason:
        querystring['reason'] = reason
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-songkick.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_search(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for locations: a city and its metro area. A metro area is a city or a collection of cities that Songkick uses to notify users of concerts near them."
    
    """
    url = f"https://community-songkick.p.rapidapi.com/search/locations.json"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-songkick.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def metro_area_s_upcoming_events_calendar(metro_area_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find upcoming events for a metro area. A metro area is a city or a collection of cities that Songkick uses to notify users of concerts near them. Use our location search to find a metro area ID."
    
    """
    url = f"https://community-songkick.p.rapidapi.com/metro_areas/{metro_area_id}/calendar.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-songkick.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upcoming_event_search(artist_name: str=None, location: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-songkick.p.rapidapi.com/events.json"
    querystring = {}
    if artist_name:
        querystring['artist_name'] = artist_name
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-songkick.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

