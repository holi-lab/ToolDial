import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_location_by_name_or_zip_code(city: str='New York', lang: str='en', zipcode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search location by Name or zip code and get the key for the forecast"
    city: name of a city
        lang: language [en,fr,nl,es,pt,it,tr,gr,cz,pl,ru,cn]
        zipcode: zip code of the city
        
    """
    url = f"https://climate-data.p.rapidapi.com/api/getlocation"
    querystring = {}
    if city:
        querystring['CITY'] = city
    if lang:
        querystring['LANG'] = lang
    if zipcode:
        querystring['ZIPCODE'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_cities_in_one_country(country: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of cities in one Country"
    country: Country Code
        lang: Language [en,fr,pl,pt,es,it,gr,tr,ru,cn,cz]
        
    """
    url = f"https://climate-data.p.rapidapi.com/api/countrycitylist"
    querystring = {'COUNTRY': country, }
    if lang:
        querystring['LANG'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_countries(lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all Countries"
    lang: Language [en,fr,de,nl,es,pt,pl,tr,gr,tr,cn]
        
    """
    url = f"https://climate-data.p.rapidapi.com/api/countrycitylist"
    querystring = {}
    if lang:
        querystring['LANG'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_climate_data_by_lat_lon_or_key(lat: int=45, lon: int=-70, key: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get climate for  the location Lat/Lon"
    lat: latitude
        lon: longitude
        key: key of city from List of cities
        lang: Language [en,de,nl,fr,pl,gr,it,cn,ru,cz,pt,es]
        
    """
    url = f"https://climate-data.p.rapidapi.com/api/getclimatedata"
    querystring = {}
    if lat:
        querystring['LAT'] = lat
    if lon:
        querystring['LON'] = lon
    if key:
        querystring['KEY'] = key
    if lang:
        querystring['LANG'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "climate-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

