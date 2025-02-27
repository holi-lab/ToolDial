import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def forecast_daily(geocode: str, units: str, days: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The forecast API returns the geocode weather forecasts for the current day up to the endpoint duration in days."
    geocode: latitude,longitude
        units: e=Imperial (English), m=Metric, h=Hybrid (UK), s=Metric SI
        
    """
    url = f"https://weather-com.p.rapidapi.com/v3/wx/forecast/daily/{days}"
    querystring = {'geocode': geocode, 'units': units, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forecast_hourly(geocode: str, units: str, days: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The forecast API returns the geocode weather forecasts for the current day up to the endpoint duration in hours."
    geocode: latitude,longitude
        units: e=Imperial (English), m=Metric, h=Hybrid (UK), s=Metric SI
        
    """
    url = f"https://weather-com.p.rapidapi.com/v3/wx/forecast/hourly/{days}"
    querystring = {'geocode': geocode, 'units': units, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_30_days(geocode: str, units: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Historical Conditions are generated from The Weather Company (TWC) Currents On Demand (COD) system. The API provides information on temperature, precipitation and other related weather observations elements as well as date/time, weather icon codes and phrases."
    geocode: latitude,longitude
        units: e=Imperial (English), m=Metric, h=Hybrid (UK), s=Metric SI
        
    """
    url = f"https://weather-com.p.rapidapi.com/v3/wx/conditions/historical/dailysummary/30day"
    querystring = {'geocode': geocode, 'units': units, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def covid19(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to view the current state of a disease for a set of countries globally"
    
    """
    url = f"https://weather-com.p.rapidapi.com/v3/wx/disease/tracker/countryList/current"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

