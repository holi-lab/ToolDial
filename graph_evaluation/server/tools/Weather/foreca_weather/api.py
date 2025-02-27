import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nowcast(location: str, dataset: str='full', windunit: str='MS', tz: str='Europe/London', tempunit: str='C', alt: int=0, periods: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "3-hour forecast in 15-minute time steps."
    dataset: Variable set
        windunit: Wind speed unit in response.
        tz: Time zone in response (IANA time zone database names)
        tempunit: Temperature unit in response
        alt: Altitude (meters)
        periods: Number of time steps (default 8, maximum 12)
        
    """
    url = f"https://foreca-weather.p.rapidapi.com/forecast/15minutely/{location}"
    querystring = {}
    if dataset:
        querystring['dataset'] = dataset
    if windunit:
        querystring['windunit'] = windunit
    if tz:
        querystring['tz'] = tz
    if tempunit:
        querystring['tempunit'] = tempunit
    if alt:
        querystring['alt'] = alt
    if periods:
        querystring['periods'] = periods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foreca-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def three_hourly(location: str, history: bool=None, tz: str='Europe/London', dataset: str='full', tempunit: str='C', alt: int=0, periods: int=8, windunit: str='MS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A longer three-hourly forecast up to 14 days."
    history: Whether to include 24 hours of past data.
        tz: Time zone in response (IANA time zone database names)
        dataset: Variable set
        tempunit: Temperature unit in response
        alt: Altitude (meters)
        periods: Number of time steps (default 8, maximum 12)
        windunit: Wind speed unit in response.
        
    """
    url = f"https://foreca-weather.p.rapidapi.com/forecast/3hourly/{location}"
    querystring = {}
    if history:
        querystring['history'] = history
    if tz:
        querystring['tz'] = tz
    if dataset:
        querystring['dataset'] = dataset
    if tempunit:
        querystring['tempunit'] = tempunit
    if alt:
        querystring['alt'] = alt
    if periods:
        querystring['periods'] = periods
    if windunit:
        querystring['windunit'] = windunit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foreca-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily(location: str, alt: int=0, dataset: str='full', tempunit: str='C', windunit: str='MS', periods: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Daily forecast."
    alt: Altitude (meters)
        dataset: Variable set
        tempunit: Temperature unit in response
        windunit: Wind speed unit in response.
        periods: Number of time steps (default 8, maximum 12)
        
    """
    url = f"https://foreca-weather.p.rapidapi.com/forecast/daily/{location}"
    querystring = {}
    if alt:
        querystring['alt'] = alt
    if dataset:
        querystring['dataset'] = dataset
    if tempunit:
        querystring['tempunit'] = tempunit
    if windunit:
        querystring['windunit'] = windunit
    if periods:
        querystring['periods'] = periods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foreca-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_info(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Metadata for location."
    
    """
    url = f"https://foreca-weather.p.rapidapi.com/location/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foreca-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current(location: str, windunit: str='MS', alt: int=0, lang: str='en', tz: str='Europe/London', tempunit: str='C', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current weather estimate for location."
    windunit: Wind speed unit in response.
        alt: Altitude (meters)
        lang: Language (ISO 639-1 codes). Options: de, en, es, fr, it, pl, ru, fi, sv, nl, ko, pt, th, tr, zh, zh_TW (Chinese in Taiwan), zh_CN (Chinese in China). (default en)
        tz: Time zone in response (IANA time zone database names)
        tempunit: Temperature unit in response
        
    """
    url = f"https://foreca-weather.p.rapidapi.com/current/{location}"
    querystring = {}
    if windunit:
        querystring['windunit'] = windunit
    if alt:
        querystring['alt'] = alt
    if lang:
        querystring['lang'] = lang
    if tz:
        querystring['tz'] = tz
    if tempunit:
        querystring['tempunit'] = tempunit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foreca-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_observations(location: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Observations from nearby representative weather stations."
    
    """
    url = f"https://foreca-weather.p.rapidapi.com/observation/latest/{location}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foreca-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_search(query: str, lang: str='en', country: str='in', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for locations by name."
    
    """
    url = f"https://foreca-weather.p.rapidapi.com/location/search/{query}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foreca-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourly(location: str, alt: int=0, history: bool=None, dataset: str='full', tz: str='Europe/London', periods: int=8, windunit: str='MS', tempunit: str='C', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hourly forecast."
    alt: Altitude (meters)
        history: Whether to include 24 hours of past data.
        dataset: Variable set
        tz: Time zone in response (IANA time zone database names)
        periods: Number of time steps (default 8, maximum 12)
        windunit: Wind speed unit in response.
        tempunit: Temperature unit in response
        
    """
    url = f"https://foreca-weather.p.rapidapi.com/forecast/hourly/{location}"
    querystring = {}
    if alt:
        querystring['alt'] = alt
    if history:
        querystring['history'] = history
    if dataset:
        querystring['dataset'] = dataset
    if tz:
        querystring['tz'] = tz
    if periods:
        querystring['periods'] = periods
    if windunit:
        querystring['windunit'] = windunit
    if tempunit:
        querystring['tempunit'] = tempunit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foreca-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

