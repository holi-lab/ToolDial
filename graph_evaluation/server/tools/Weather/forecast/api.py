import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See the current API version. Here is the link to the documentation"
    
    """
    url = f"https://forecast9.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rapidapigetforecasthourlybycoordinates(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get forecast hourly information."
    latitude: Latitude
        longitude: Longitude
        
    """
    url = f"https://forecast9.p.rapidapi.com/rapidapi/forecast/{latitude}/{longitude}/hourly/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rapidapigetforecastsummarybycoordinates(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Forecast summary information."
    longitude: Longitude
        latitude: Latitude
        
    """
    url = f"https://forecast9.p.rapidapi.com/rapidapi/forecast/{latitude}/{longitude}/summary/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rapidapigetobservationhistorybystationid(is_id: str, date: str='2020-05-27', range: str='48hours', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an History object with historical information."
    id: Station ID
        date: Show historical data from specific date
        range: Range of historical data
        
    """
    url = f"https://forecast9.p.rapidapi.com/rapidapi/station/{is_id}/history/"
    querystring = {}
    if date:
        querystring['date'] = date
    if range:
        querystring['range'] = range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rapidapigetobservationhistorybycoordinates(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an Observation object with historical information."
    longitude: Longitude
        latitude: Latitude
        
    """
    url = f"https://forecast9.p.rapidapi.com/rapidapi/station/nearBy/{latitude}/{longitude}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if the API is running."
    
    """
    url = f"https://forecast9.p.rapidapi.com/status/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rapidapigetforecastsummarybylocationname(locationname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Forecast"
    locationname: Location Name
        
    """
    url = f"https://forecast9.p.rapidapi.com/rapidapi/forecast/{locationname}/summary/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rapidapigethourlyforecastbylocationname(locationname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    locationname: Location Name
        
    """
    url = f"https://forecast9.p.rapidapi.com/rapidapi/forecast/{locationname}/hourly/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

