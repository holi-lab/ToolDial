import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def by_coordinates(lat: int, lng: int, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearest places readings based around the given latitude and longitude"
    lat: Specifies latitude
        lng: Specifies longitude
        limit: Limits the number of rows returned
        
    """
    url = f"https://ambee-air-quality.p.rapidapi.com/latest/by-lat-lng"
    querystring = {'lat': lat, 'lng': lng, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-air-quality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_postal_code(postalcode: int, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check air quality for your postal code"
    postalcode: Postal code to display the air quality for
        limit: The number of rows to be displayed
        
    """
    url = f"https://ambee-air-quality.p.rapidapi.com/latest/by-postal-code"
    querystring = {'postalCode': postalcode, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-air-quality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_city(city: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest air quality data for your city"
    city: The name of the city for which you want the air quality
        limit: The number of rows to be returned
        
    """
    url = f"https://ambee-air-quality.p.rapidapi.com/latest/by-city"
    querystring = {'city': city, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-air-quality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def by_country(countrycode: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the air quality data for a country"
    countrycode: The international country code for the country you want the air quality for
        limit: The number of rows to be returned
        
    """
    url = f"https://ambee-air-quality.p.rapidapi.com/latest/by-country-code"
    querystring = {'countryCode': countrycode, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-air-quality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather(lat: int, lng: int, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weather data for any place based on coordinates"
    lat: Latitude of the coordinate to get the weather
        lng: Longitude of the coordinate to find the weather
        limit: The number of rows to be returned
        
    """
    url = f"https://ambee-air-quality.p.rapidapi.com/weather/by-lat-lng"
    querystring = {'lat': lat, 'lng': lng, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ambee-air-quality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

