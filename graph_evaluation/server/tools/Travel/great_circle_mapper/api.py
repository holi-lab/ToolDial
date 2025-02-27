import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def aircraft_type_read(icao_iata: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get aircraft type data by IATA or ICAO code"
    icao_iata: ICAO 3 letter or IATA 2 letter code
        
    """
    url = f"https://greatcirclemapper.p.rapidapi.com/aircraft/read/{icao_iata}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greatcirclemapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def air_route_calculation(route: str, speed: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate distance and flight time for any airports and any speed"
    route: ICAO airport cides, separated by hyphens
        speed: Speed in kts
        
    """
    url = f"https://greatcirclemapper.p.rapidapi.com/airports/route/{route}/{speed}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greatcirclemapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get a list of airport records"
    query: ICAO code, IATA code, town, airport name
        
    """
    url = f"https://greatcirclemapper.p.rapidapi.com/airports/search/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greatcirclemapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_read(icao_iata: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get airport by IATA code or ICAO code"
    icao_iata: ICAO code or IATA code
        
    """
    url = f"https://greatcirclemapper.p.rapidapi.com/airports/read/{icao_iata}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greatcirclemapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

