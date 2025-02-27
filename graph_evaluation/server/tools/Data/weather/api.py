import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_5_day_forecast(lat: int, lon: int, units: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "3 hour interval - 5 day forecast for a given lat/lon"
    lat: Latitude (Degrees)
        lon: Longitude (degrees)
        units: I = Imperial, S = Scientific, M = Metric (Default)
        lang: Language for weather condition
        
    """
    url = f"https://weatherbit-v1-mashape.p.rapidapi.com/forecast/3hourly"
    querystring = {'lat': lat, 'lon': lon, }
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherbit-v1-mashape.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_weather_data_of_a_location(lat: int, lon: int, lang: str=None, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current (most recent) weather observation of a given location"
    lat: Latitude
        lon: Longitude
        lang: Language
        units: I = Imperial, S = Scientific, M = Metric (Default)
        
    """
    url = f"https://weatherbit-v1-mashape.p.rapidapi.com/current"
    querystring = {'lat': lat, 'lon': lon, }
    if lang:
        querystring['lang'] = lang
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherbit-v1-mashape.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_hour_minutely_forecast_nowcast(lat: int, lon: int, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a 60 minute "Nowcast" for precipitation, and snowfall"
    units: metric (Default), imperial
        
    """
    url = f"https://weatherbit-v1-mashape.p.rapidapi.com/forecast/minutely"
    querystring = {'lat': lat, 'lon': lon, }
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherbit-v1-mashape.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_16_day_forecast(lon: int, lat: int, lang: str=None, units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a 16 day (daily) forecast"
    lon: Longitude
        lat: Latitude
        lang: Language for weather description
        units: metric (Default), imperial
        
    """
    url = f"https://weatherbit-v1-mashape.p.rapidapi.com/forecast/daily"
    querystring = {'lon': lon, 'lat': lat, }
    if lang:
        querystring['lang'] = lang
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherbit-v1-mashape.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_120_hour_forecast(lat: int, lon: int, lang: str=None, hours: str='48', units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a forecast for up to 120 hours in the future (default 48 hours)"
    lat: Latitude
        lon: Longitude
        lang: Language for weather description
        hours: Limit number of forecast hours
        units: I = Imperial, S = Scientific, M = Metric (Default)
        
    """
    url = f"https://weatherbit-v1-mashape.p.rapidapi.com/forecast/hourly"
    querystring = {'lat': lat, 'lon': lon, }
    if lang:
        querystring['lang'] = lang
    if hours:
        querystring['hours'] = hours
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherbit-v1-mashape.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def severe_weather_alerts(lat: int, lon: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get servere weather alerts from local meteorlogical agencies (US, EU, and Isreal supported )"
    lat: Latitude
        lon: Longitude
        
    """
    url = f"https://weatherbit-v1-mashape.p.rapidapi.com/alerts"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weatherbit-v1-mashape.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

