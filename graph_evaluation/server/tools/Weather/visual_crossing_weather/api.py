import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weather_forecast_data(aggregatehours: int, location: str, contenttype: str='csv', unitgroup: str='us', shortcolumnnames: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides access to weather forecast information. The forecast is available for up to seven days at the hourly, 12 hour and daily summary level."
    aggregatehours: The interval between weather forecast data in the output. 1 represents an hourly forecast, 24 represents a daily forecast. As the source data is calculated at the hourly level, records calculated at 12 or 24 hours are aggregated to indicate the predominant weather condition during that time period. For example the maximum temperature, total precipitation, maximum windspeed etc.  Supported values 1,12 or 24.
        location: he address or latitude or longitude of the location. Addresses can be specified as full addresses. The system will also attempt to match partial addresses such as city, state, zip code, postal code and other common formats. When specify a point based on longitude, latitude, the format must be specifed as latitude,longitude where both latitude and longitude are in decimal degrees. latitude should run from -90 to 90 and longitude from -180 to 180 (with 0 being at the prime meridian through London, UK).
        contenttype: When present, choose between json or csv output
        unitgroup: unitGroup - The system of units used for the output data.  Supported values are us,uk,metric.
        shortcolumnnames:  When false, the returned dataset includes descriptive column names. When true, returns shorter, abbreviated column names with only alphanumeric characters. The short names are useful for programmatic use of the data.
        
    """
    url = f"https://visual-crossing-weather.p.rapidapi.com/forecast"
    querystring = {'aggregateHours': aggregatehours, 'location': location, }
    if contenttype:
        querystring['contentType'] = contenttype
    if unitgroup:
        querystring['unitGroup'] = unitgroup
    if shortcolumnnames:
        querystring['shortColumnNames'] = shortcolumnnames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "visual-crossing-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_weather_record(location: str, enddatetime: str, startdatetime: str, aggregatehours: int, unitgroup: str, contenttype: str='csv', dayendtime: str='17:00:00', daystarttime: str='8:00:00', shortcolumnnames: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The weather history data is suitable for retrieving hourly or daily historical weather records."
    location: The address or latitude or longitude of the location. Addresses can be specified as full addresses. The system will also attempt to match partial addresses such as city, state, zip code, postal code and other common formats. When specify a point based on longitude, latitude, the format must be specifed as latitude,longitude where both latitude and longitude are in decimal degrees. latitude should run from -90 to 90 and longitude from -180 to 180 (with 0 being at the prime meridian through London, UK).
        enddatetime: The date time for the start of the data request using the time zone of the location. In the ISO format: yyyy-MM-ddTHH:mm:ss. Hours should be specified in 24 hour clock format.
        startdatetime: The date time for the start of the data request using the time zone of the location. In the ISO format: yyyy-MM-ddTHH:mm:ss. Hours should be specified in 24 hour clock format.
        aggregatehours: The interval between weather history data in the output. 1 represent hourly records, 24 represents a daily forecast. As the source data is recorded at the hourly level, 24 hour records are aggregated to indicate the predominant weather conditions during that time period.  Supported values 1 or 24.
        unitgroup: The system of units used for the output data.  Supported values are us,uk,metric
        contenttype: When present, choose between json or csv output
        dayendtime: When present and not set to the same as the dayEndTime, filters the output to records that between the specified day times.
        daystarttime: When present and not set to the same as the dayEndTime, filters the output to records that between the specified day times. This is useful for setting filters for business hours. Format h:m:ss (eg 9:00:00 woudl be 9am).
        shortcolumnnames: When false, the returned dataset includes descriptive column names. When true, returns shorter, abbreviated column names with only alphanumeric characters. The short names are useful for programmatic use of the data.
        
    """
    url = f"https://visual-crossing-weather.p.rapidapi.com/history"
    querystring = {'location': location, 'endDateTime': enddatetime, 'startDateTime': startdatetime, 'aggregateHours': aggregatehours, 'unitGroup': unitgroup, }
    if contenttype:
        querystring['contentType'] = contenttype
    if dayendtime:
        querystring['dayEndTime'] = dayendtime
    if daystarttime:
        querystring['dayStartTime'] = daystarttime
    if shortcolumnnames:
        querystring['shortColumnNames'] = shortcolumnnames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "visual-crossing-weather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

