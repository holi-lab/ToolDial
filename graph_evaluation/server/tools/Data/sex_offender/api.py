import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def by_state_city_name(name: str, city: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Sex Offenders by State & City & Sex Offender Name"
    name: Prefix of a sex offender name, e.g., first name 'David' for 'David Wayne Todd'
        city: City name, e.g., Dayton
        state: State full name (e.g., Ohio) or abbreviation (e.g., OH) 
        
    """
    url = f"https://sex-offender.p.rapidapi.com/state-city-name"
    querystring = {'name': name, 'city': city, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sex-offender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_location(radius: str, lat: str, lng: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Sex Offenders by Location Coordinate and Radius"
    radius: Radius in mile (10 miles maximum)
        lat: Location latitude
        lng: Location longitude
        
    """
    url = f"https://sex-offender.p.rapidapi.com/location"
    querystring = {'radius': radius, 'lat': lat, 'lng': lng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sex-offender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_zip_code_name(zipcode: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Sex Offenders by Zip Code & Sex Offender Name"
    zipcode: Zip code
        name: Prefix of a sex offender name, e.g., first name 'David' for 'David Wayne Todd'
        
    """
    url = f"https://sex-offender.p.rapidapi.com/zipcode-name"
    querystring = {'zipcode': zipcode, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sex-offender.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

