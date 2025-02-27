import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def daily_forecast_5_days(longitude: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a 5-day daily forecast for the given latitude and longitude."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/daily/5"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourly_forecast_240_hours(longitude: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a 240-hour forecast for the given latitude and longitude."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/hourly/full"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourly_forecast_48_hours(latitude: str, longitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a 48-hour forecast for the given latitude and longitude."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/hourly/48"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_daily(longitude: str, date: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a historical daily forecast for the given latitude, longitude, and start date/time."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/historical/daily"
    querystring = {'longitude': longitude, 'date': date, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_hourly(longitude: str, date: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a historical hourly forecast for the given latitude, longitude, and start date/time."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/historical/hourly"
    querystring = {'longitude': longitude, 'date': date, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_forecast_10_days(longitude: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a 10-day daily forecast for the given latitude and longitude."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/daily/full"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_conditions_basic(longitude: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get (basic) current conditions for the given latitude and longitude."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/current/basic"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def complete_weather_single_call(longitude: str, latitude: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current conditions, hourly forecast, daily forecast, and alerts in a single response for the given latitude and longitude (you must supply ISO country code to receive weather alerts)."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/all/full"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_conditions_detailed(longitude: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get (detailed) current conditions for the given latitude and longitude."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/current/detail"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather_alerts(country: str, longitude: str, latitude: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weather alerts posted by the local meteorological agency for the given latitude and longitude."
    
    """
    url = f"https://easy-weather1.p.rapidapi.com/alerts/full"
    querystring = {'country': country, 'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-weather1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

