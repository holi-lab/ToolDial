import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def positions_of_the_sun_by_coordinates(lon: int, date: str, lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides the times of various sun positions based on the coordinates and date specified in the query parameters."
    
    """
    url = f"https://sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com/bycoordinates"
    querystring = {'lon': lon, 'date': date, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def positions_of_the_sun_by_city(date: str, city: str, countrycode: str, statecode: str='bu', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides the times of various sun positions based on the city and date specified in the query parameters.
		The state code is not mandatory but highly recommended due to the presence of cities with the same name. Specifying the state code helps to avoid any confusion and ensures more accurate results."
    
    """
    url = f"https://sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com/bycity"
    querystring = {'date': date, 'city': city, 'countrycode': countrycode, }
    if statecode:
        querystring['statecode'] = statecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This response contains the names and codes of the countries."
    
    """
    url = f"https://sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com/allcountries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def states_by_country_code(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint responds with the names and codes of all states or provinces belonging to the country specified by the "countrycode" query parameter."
    
    """
    url = f"https://sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com/states-by-countrycode"
    querystring = {'countrycode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cities_by_country_code_and_state_code(countrycode: str, statecode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint responds with the names and codes of all cities belonging to the state corresponding to the "statecode" query parameter. It is important to note that both the "countrycode" and "statecode" query parameters are required for a successful API request."
    
    """
    url = f"https://sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com/cities-by-countrycode-and-statecode"
    querystring = {'countrycode': countrycode, 'statecode': statecode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunset-time-sunrise-time-by-city-rest-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

