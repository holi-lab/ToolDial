import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_airports_by_coordinates(lon: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Get Airports By Coordinates" API enables you to retrieve a list of airports based on specific latitude and longitude coordinates. By providing the latitude and longitude values, this API returns comprehensive details about airports located within that vicinity. This information includes airport names, IATA codes, locations, and additional relevant data."
    
    """
    url = f"https://airports-finder1.p.rapidapi.com/airports/coordinates/{lat}/{lon}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airports-finder1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airports_by_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Get Airports By Country" API allows you to easily retrieve a list of airports based on a specific country. This API provides comprehensive details about airports, including their names, IATA codes, locations, and additional information."
    
    """
    url = f"https://airports-finder1.p.rapidapi.com/airports/country/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airports-finder1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airports_by_city(city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Get Airports By City" API enables you to effortlessly retrieve a list of airports based on a specific city. This API provides comprehensive details about airports, including their names, IATA codes, locations, and additional information."
    
    """
    url = f"https://airports-finder1.p.rapidapi.com/airports/city/{city}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airports-finder1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airport_details_by_code(code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Get Airport Details By Code" API allows you to quickly retrieve comprehensive information about airports based on their IATA codes. Obtain precise details such as airport names, locations, time zones, and more with this efficient and reliable API."
    
    """
    url = f"https://airports-finder1.p.rapidapi.com/airports/{code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airports-finder1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

