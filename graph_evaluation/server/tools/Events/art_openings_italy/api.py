import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def live_events(city: str='Firenze', genre: str='festival', location: str='PIAZZA SANTA MARIA NOVELLA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To call this service, you would make a GET request to the endpoint /api/ongoing-events with the optional query parameter city. When the service is called, it retrieves a list of events from the database, filtered by the city parameter if it is provided."
    
    """
    url = f"https://art-openings-italy.p.rapidapi.com/api/ongoing-events"
    querystring = {}
    if city:
        querystring['city'] = city
    if genre:
        querystring['genre'] = genre
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "art-openings-italy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This service returns all data related to a specific id. The id field in the example you provided is a unique identifier for the event. It is a string of characters that follows the format of a universally unique identifier (UUID), which is a standardized way of generating a 128-bit identifier that is guaranteed to be unique across all devices and all time."
    
    """
    url = f"https://art-openings-italy.p.rapidapi.com/api/events/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "art-openings-italy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health_check(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The service it allows you to monitor if the application is up time. It returns an healthcheck object that has three properties uptime , message and timestamp."
    
    """
    url = f"https://art-openings-italy.p.rapidapi.com/api/healthcheck"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "art-openings-italy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_genres(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This service returns a list of all genres of the events."
    
    """
    url = f"https://art-openings-italy.p.rapidapi.com/api/genres"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "art-openings-italy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_locations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This service returns a list of all locations where art events take place."
    
    """
    url = f"https://art-openings-italy.p.rapidapi.com/api/locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "art-openings-italy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_cities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This service returns a list of all cities where art events take place."
    
    """
    url = f"https://art-openings-italy.p.rapidapi.com/api/cities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "art-openings-italy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

