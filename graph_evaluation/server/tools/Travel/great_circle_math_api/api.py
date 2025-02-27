import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_distance_by_city_state_country_in_km(state2: str, country2: str, country1: str, state1: str, city1: str, city2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes city, state, and country of both locations and returns latitude, longitude, and calculated miles in kilometers."
    
    """
    url = f"https://great-circle-math-api.p.rapidapi.com/byLocation/byCityStateCountryKm/{city1}/{state1}/{country1}/{city2}/{state2}/{country2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "great-circle-math-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_distance_by_city_state_country(country1: str, country2: str, state2: str, city2: str, city1: str, state1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes city, state, and country of both locations and returns latitude, longitude, and calculated miles."
    
    """
    url = f"https://great-circle-math-api.p.rapidapi.com/byLocation/byCityStateCountry/{city1}/{state1}/{country1}/{city2}/{state2}/{country2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "great-circle-math-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_distance(latb: int, longa: int, lata: int, longb: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the circle math distance in miles."
    
    """
    url = f"https://great-circle-math-api.p.rapidapi.com/calculate/getDistance/{lata}/{longa}/{latb}/{longb}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "great-circle-math-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_distance_in_km(latb: int, longb: int, longa: int, lata: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns circle math distance in kilometers."
    
    """
    url = f"https://great-circle-math-api.p.rapidapi.com/calculate/getDistanceInKm/{lata}/{longa}/{latb}/{longb}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "great-circle-math-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

