import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def by_zip_code(zipcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Homeless Shelters by Zip Code"
    zipcode: Zip code
        
    """
    url = f"https://homeless-shelter.p.rapidapi.com/zipcode"
    querystring = {'zipcode': zipcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "homeless-shelter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_state_city(state: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Homeless Shelters by State & City"
    state: State full name (e.g., Washington) or abbreviation (e.g., WA) 
        city: City name, e.g., Bellevue
        
    """
    url = f"https://homeless-shelter.p.rapidapi.com/state-city"
    querystring = {'state': state, 'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "homeless-shelter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_location(lat: str, radius: str, lng: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Homeless Shelters by Location Coordinate and Radius"
    lat: Location latitude
        radius: Radius in mile (10 miles maximum)
        lng: Location longitude
        
    """
    url = f"https://homeless-shelter.p.rapidapi.com/location"
    querystring = {'lat': lat, 'radius': radius, 'lng': lng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "homeless-shelter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

