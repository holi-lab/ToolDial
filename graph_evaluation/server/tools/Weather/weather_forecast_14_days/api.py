import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getforecastdata(key: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get forecast for 14 days for the location "KEY""
    key: key for city, key get from getcitykey
        lang: Language [en,de,nl,fr,pl,gr,it,cn,ru,cz,pt,es]
        
    """
    url = f"https://weather-forecast-14-days.p.rapidapi.com/api/getforecastdata"
    querystring = {'KEY': key, }
    if lang:
        querystring['LANG'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-forecast-14-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_forecastdata_by_lat_lon(lat: int, lon: int, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get forecast for 14 days for the location Lat/Lon"
    lat: Latitude
        lon: Longitude
        lang: Language [en,de,nl,fr,pl,gr,it,cn,ru,cz,pt,es]
        
    """
    url = f"https://weather-forecast-14-days.p.rapidapi.com/api/getforecastdata"
    querystring = {'LAT': lat, 'LON': lon, }
    if lang:
        querystring['LANG'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-forecast-14-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_all_countries(lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all Countries"
    lang: Language [en,fr,de,nl,es,pt,pl,tr,gr,tr,cn]
        
    """
    url = f"https://weather-forecast-14-days.p.rapidapi.com/api/countrycitylist"
    querystring = {}
    if lang:
        querystring['LANG'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-forecast-14-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_cities_in_one_country(lang: str='en', country: str='UK', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of cities in one Country"
    lang: Language [en,fr,pl,pt,es,it,gr,tr,ru,cn,cz]
        country: Country
        
    """
    url = f"https://weather-forecast-14-days.p.rapidapi.com/api/countrycitylist"
    querystring = {}
    if lang:
        querystring['LANG'] = lang
    if country:
        querystring['COUNTRY'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-forecast-14-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_location_by_name_or_zip_code(zipcode: str=None, lang: str='en', city: str='New York', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search location by Name or zip code and get the key for the forecast"
    zipcode: zip code of the city
        lang: language [en,fr,nl,es,pt,it,tr,gr,cz,pl,ru,cn]
        city: name of a city
        
    """
    url = f"https://weather-forecast-14-days.p.rapidapi.com/api/getlocation"
    querystring = {}
    if zipcode:
        querystring['ZIPCODE'] = zipcode
    if lang:
        querystring['LANG'] = lang
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-forecast-14-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

