import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def locations_get_details(placeid: str, language: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of specific location"
    placeid: The value of placeid field returned in .../locations/search endpoint
        language: The language code
        
    """
    url = f"https://weather338.p.rapidapi.com/locations/get-details"
    querystring = {'placeid': placeid, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather338.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_search(query: str, language: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search City or Zipcode"
    query: City name or Postal code
        language: The language code
        
    """
    url = f"https://weather338.p.rapidapi.com/locations/search"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather338.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather_forecast(longitude: int, date: str, latitude: int, language: str='en-US', units: str='m', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weather forecast information to specific GEO location"
    longitude: The longitude of GEO location to get weather forecast
        date: Date to get weather forecast, format as yyyyMMdd
        latitude: The latitude of GEO location to get weather forecast
        language: The language code
        units: One of the followings : e - English | m - Metric | h - Hybrid
        
    """
    url = f"https://weather338.p.rapidapi.com/weather/forecast"
    querystring = {'longitude': longitude, 'date': date, 'latitude': latitude, }
    if language:
        querystring['language'] = language
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather338.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list(limit: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest weather forecast news"
    limit: Use for paging purpose
        offset: Use for paging purpose
        
    """
    url = f"https://weather338.p.rapidapi.com/news/list"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather338.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

