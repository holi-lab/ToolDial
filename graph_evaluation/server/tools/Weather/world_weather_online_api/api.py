import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def marine_history_weather_api(q: str, date: str, enddate: str='2020-05-26', format: str=None, tide: str='no', tp: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Historical Marine Weather API method allows you to access marine data since 1st Jan, 2015 for a given longitude and latitude, as well as tide data. The Historical Marine Weather API returns weather elements such as temperature, precipitation (rainfall), weather description, weather icon, wind speed, Tide data, significant wave height, swell height, swell direction and swell period."
    
    """
    url = f"https://world-weather-online-api1.p.rapidapi.com/past-marine.ashx"
    querystring = {'q': q, 'date': date, }
    if enddate:
        querystring['enddate'] = enddate
    if format:
        querystring['format'] = format
    if tide:
        querystring['tide'] = tide
    if tp:
        querystring['tp'] = tp
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-weather-online-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_api(q: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Location search API method takes a query value and returns information about the location, including area name, country, latitude/longitude, population, and a URL for the weather information."
    
    """
    url = f"https://world-weather-online-api1.p.rapidapi.com/search.ashx"
    querystring = {'q': q, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-weather-online-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def local_weather_api(q: str, tp: str=None, aqi: str='yes', format: str=None, lang: str='en', alerts: str='no', num_of_days: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Local Weather REST API (also called City and Town Weather API) method allows you to access current weather conditions, the next 14 days of accurate and reliable weather forecast, Air Quality Data, Weather Alerts and Monthly Climate Averages for over 4 million cities and towns worldwide. The Local Weather API returns weather elements such as temperature, precipitation (rainfall), weather description, weather icon, wind speed, etc."
    
    """
    url = f"https://world-weather-online-api1.p.rapidapi.com/weather.ashx"
    querystring = {'q': q, }
    if tp:
        querystring['tp'] = tp
    if aqi:
        querystring['aqi'] = aqi
    if format:
        querystring['format'] = format
    if lang:
        querystring['lang'] = lang
    if alerts:
        querystring['alerts'] = alerts
    if num_of_days:
        querystring['num_of_days'] = num_of_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-weather-online-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def local_history_weather_api(date: str, q: str, enddate: str='2015-05-31', tp: str=None, format: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Local Historical or Past Weather API (also known as City and Town Historical Weather API) allows you to access weather conditions from 1st July 2008 up until the present time. The API returns weather elements such as temperature, precipitation (rainfall), weather description, weather icon and wind speed."
    
    """
    url = f"https://world-weather-online-api1.p.rapidapi.com/past-weather.ashx"
    querystring = {'date': date, 'q': q, }
    if enddate:
        querystring['enddate'] = enddate
    if tp:
        querystring['tp'] = tp
    if format:
        querystring['format'] = format
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-weather-online-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_zone_api(q: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Time Zone API method retrieves current local time and UTC offset hour and minute for a specified location."
    
    """
    url = f"https://world-weather-online-api1.p.rapidapi.com/tz.ashx"
    querystring = {'q': q, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-weather-online-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def astronomy_api(q: str, date: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Astronomy and Lunar API method allows you to access astronomy information for any given date."
    
    """
    url = f"https://world-weather-online-api1.p.rapidapi.com/astronomy.ashx"
    querystring = {'q': q, 'date': date, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-weather-online-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ski_weather_api(q: str, tp: str=None, lang: str='en', format: str=None, num_of_days: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ski Weather API provides worldwide ski and mountain weather forecast for upto 7-days."
    
    """
    url = f"https://world-weather-online-api1.p.rapidapi.com/ski.ashx"
    querystring = {'q': q, }
    if tp:
        querystring['tp'] = tp
    if lang:
        querystring['lang'] = lang
    if format:
        querystring['format'] = format
    if num_of_days:
        querystring['num_of_days'] = num_of_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-weather-online-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def marine_weather_api(q: str, tp: str=None, format: str=None, lang: str='en', tide: str='no', num_of_days: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Marine Weather API provides worldwide marine weather forecast and tide data for upto 7-days."
    
    """
    url = f"https://world-weather-online-api1.p.rapidapi.com/marine.ashx"
    querystring = {'q': q, }
    if tp:
        querystring['tp'] = tp
    if format:
        querystring['format'] = format
    if lang:
        querystring['lang'] = lang
    if tide:
        querystring['tide'] = tide
    if num_of_days:
        querystring['num_of_days'] = num_of_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-weather-online-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

