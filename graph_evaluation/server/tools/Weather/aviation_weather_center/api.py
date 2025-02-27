import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def station_info(datasource: str, stationstring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about a weather reporting station."
    
    """
    url = f"https://aviation-weather-center.p.rapidapi.com/adds/dataserver_current/httpparam"
    querystring = {'datasource': datasource, 'stationString': stationstring, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aviation-weather-center.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_recent_tafs(mostrecentforeachstation: str=None, datasource: str='tafs', stationstring: str='KSFO', mostrecent: bool=None, hoursbeforenow: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain the most recent TAFs from the past X hours. If you use the `mostRecent` flag only one TAF is returned. Otherwise, all TAFs from the specified time window will be returned."
    mostrecentforeachstation: Used to get the most recent value for EACH station when multiple stations are specified

- **constraint** - request the most recent for each METAR station in the fastest fashion. Not appropriate for historical data retrieval
- **postfilter** - post filter results after applying all other constraints
- **true** - same as 'postfilter' method
- **false** - don't get the most recent for each METAR station, equivalent to omitting this parameter
        stationstring: Specify a station as a four character ICAO code (example: `KSFO`).
Can specify multiple comma separated stations (example: `KSFO,KSQL,KSNS`).
        
    """
    url = f"https://aviation-weather-center.p.rapidapi.com/adds/dataserver_current/httpparam"
    querystring = {}
    if mostrecentforeachstation:
        querystring['mostRecentForEachStation'] = mostrecentforeachstation
    if datasource:
        querystring['datasource'] = datasource
    if stationstring:
        querystring['stationString'] = stationstring
    if mostrecent:
        querystring['mostRecent'] = mostrecent
    if hoursbeforenow:
        querystring['hoursBeforeNow'] = hoursbeforenow
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aviation-weather-center.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_recent_metars(datasource: str, mostrecentforeachstation: str=None, hoursbeforenow: int=1, mostrecent: bool=None, stationstring: str='KSQL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain the most recent METARs from the past X hours. If you use the `mostRecent` flag only one METAR is returned. Otherwise, all METARs from the specified time window will be returned."
    mostrecentforeachstation: Used to get the most recent value for EACH station when multiple stations are specified

- **constraint** - request the most recent for each METAR station in the fastest fashion. Not appropriate for historical data retrieval
- **postfilter** - post filter results after applying all other constraints
- **true** - same as 'postfilter' method
- **false** - don't get the most recent for each METAR station, equivalent to omitting this parameter
        stationstring: Specify a station as a four character ICAO code (example: `KSFO`).
Can specify multiple comma separated stations (example: `KSFO,KSQL,KSNS`).
        
    """
    url = f"https://aviation-weather-center.p.rapidapi.com/adds/dataserver_current/httpparam"
    querystring = {'datasource': datasource, }
    if mostrecentforeachstation:
        querystring['mostRecentForEachStation'] = mostrecentforeachstation
    if hoursbeforenow:
        querystring['hoursBeforeNow'] = hoursbeforenow
    if mostrecent:
        querystring['mostRecent'] = mostrecent
    if stationstring:
        querystring['stationString'] = stationstring
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aviation-weather-center.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def metars_in_time_range(datasource: str, starttime: str='2021-12-15T16:48:35Z', endtime: str='2021-12-15T18:48:35Z', stationstring: str='KSQL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain all METARs for a station collected between these start and end times using ISO8601  date/time format, with UTC offset: YYYY-MM-DDThh:mm:ssZ. Please see [W3C date/time formats](http://www.w3.org/TR/NOTE-datetime)."
    
    """
    url = f"https://aviation-weather-center.p.rapidapi.com/adds/dataserver_current/httpparam"
    querystring = {'datasource': datasource, }
    if starttime:
        querystring['startTime'] = starttime
    if endtime:
        querystring['endTime'] = endtime
    if stationstring:
        querystring['stationString'] = stationstring
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aviation-weather-center.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

