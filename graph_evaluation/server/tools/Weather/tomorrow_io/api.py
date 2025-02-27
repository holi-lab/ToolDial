import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weather_forecast(location: str, timesteps: str, units: str='metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Weather forecast for Core parameters"
    
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/weather/forecast"
    querystring = {'location': location, 'timesteps': timesteps, }
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievetimelinesbasic(starttime: str, location: str, fields: str, endtime: str, timesteps: str, timezone: str=None, units: str='Metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Timeline parameters"
    starttime: Start time in ISO 8601 format \"2019-03-20T14:09:50Z\" (defaults to now)
        location: (Required) ID of a pre-defined location or latlong string
        fields: (Required) Selected fields from our data layers (polygon/polyline default to Max, if not suffix is not specified)
        endtime: End time in ISO 8601 format \"2019-03-20T14:09:50Z\" 
        timesteps: Timesteps of the timelines: \"1m\", \"5m\", \"15m\", \"30m\", \"1h\", \"1d\", and \"current\"
        timezone: Timezone of time values, according to IANA Timezone Names (defaults to UTC)
        units: Units of field values, either \"metric\" or \"imperial\" (defaults to metric)
        
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/timelines"
    querystring = {'startTime': starttime, 'location': location, 'fields': fields, 'endTime': endtime, 'timesteps': timesteps, }
    if timezone:
        querystring['timezone'] = timezone
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveweatherrecenthisotry(timesteps: str, location: str, units: str='metric', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent historical weather of a location"
    
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/weather/history/recent"
    querystring = {'timesteps': timesteps, 'location': location, }
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieverealtimeweather(location: str, timesteps: str, units: str='{{units}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Realtime Weather for a location"
    
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/weather/forecast"
    querystring = {'location': location, 'timesteps': timesteps, }
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveeventsbasic(insights: str='{{insights}}', location: str='{{location}}', buffer: str='{{buffer}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    insights: (Required) Pre-defined category names or custom insight IDs
        location: (Required) ID of a pre-defined location or latlong string
        buffer: The buffer around locations, in case of pre-defined insight categories (in km)
        
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/events"
    querystring = {}
    if insights:
        querystring['insights'] = insights
    if location:
        querystring['location'] = location
    if buffer:
        querystring['buffer'] = buffer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrievealocation(locationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    locationid: (Required) ID of a pre-defined location
        
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/locations/{locationid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listlocations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listinsights(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/insights"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveaninsight(insightid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    insightid: (Required) ID of a pre-defined insight
        
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/insights/{insightid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieveanalert(alertid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    alertid: (Required) ID of a pre-defined alert
        
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/alerts/{alertid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listalerts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/alerts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linkedlocations(alertid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    alertid: (Required) ID of a pre-defined alert
        
    """
    url = f"https://tomorrow-io1.p.rapidapi.com/v4/alerts/{alertid}/locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomorrow-io1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

