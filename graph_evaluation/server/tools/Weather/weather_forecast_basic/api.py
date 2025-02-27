import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_of_cities_in_one_country(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of cities in one Country"
    
    """
    url = f"https://weatheronline-weather-forecast-basic-v1.p.rapidapi.com/api/countrycitylist?COUNTRY={country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline-weather-forecast-basic-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all Countries"
    
    """
    url = f"https://weatheronline-weather-forecast-basic-v1.p.rapidapi.com/api/countrycitylist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline-weather-forecast-basic-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_location_by_name_or_zip_code(zipcode: str=None, city: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search location by Name or zip code and get the key for the forecast"
    zipcode: zip code of the city
        city: name of a city
        
    """
    url = f"https://weatheronline-weather-forecast-basic-v1.p.rapidapi.com/api/getlocation"
    querystring = {}
    if zipcode:
        querystring['ZIPCODE'] = zipcode
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline-weather-forecast-basic-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_forecastdata_by_lat_lon(lat: str, lon: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get forecastdata by lat/lon (without wmo)"
    lang: language [en,nl,fr,es,pt,it,gr,tr,cz,de,pl,ru,cn]
        
    """
    url = f"https://weatheronline-weather-forecast-basic-v1.p.rapidapi.com/api/getforecastdata?LAT={lat}&LON={lon}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatheronline-weather-forecast-basic-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

