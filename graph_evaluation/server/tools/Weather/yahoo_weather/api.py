import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_location(location: str, u: str='f', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by location, location is city name, e.g. location=sunnyvale,ca or location=shanghai,cn will work"
    u: f for Fahrenheit or c for Celsius, default is f
        format:  xml or json, default is json
        
    """
    url = f"https://yahoo-weather5.p.rapidapi.com/weather"
    querystring = {'location': location, }
    if u:
        querystring['u'] = u
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-weather5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_woeid(woeid: int, u: str='f', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To find your woeid, browse or search for your city from Yahoo Weather page. The woeid is in the URL for the forecast page for that city. For example, if you "Change location" for Sunnyvale, the forecast page for that city is https://www.yahoo.com/news/weather/united-states/sunnyvale/sunnyvale-**2502265**. The woeid is 2502265"
    u: f for Fahrenheit or c for Celsius, default is f
        format: xml or json, default is json
        
    """
    url = f"https://yahoo-weather5.p.rapidapi.com/weather"
    querystring = {'woeid': woeid, }
    if u:
        querystring['u'] = u
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-weather5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_geolocation_latitude_longitude(long: int, lat: int, format: str='json', u: str='f', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "input location latitude & longitude number, Example: lat=37.372&lon=-122.038"
    format: xml or json, default is json
        u: f for Fahrenheit or c for Celsius, default is f
        
    """
    url = f"https://yahoo-weather5.p.rapidapi.com/weather"
    querystring = {'long': long, 'lat': lat, }
    if format:
        querystring['format'] = format
    if u:
        querystring['u'] = u
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-weather5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

