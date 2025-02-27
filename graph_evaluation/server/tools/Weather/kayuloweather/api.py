import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weather_alert(language: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive an active weather alert."
    language: The IETF BCP 47 language tag to use for localizing responses.
        id: The unique identifier for the weather alert.
        
    """
    url = f"https://kayuloweather.p.rapidapi.com/weatherAlert/{language}/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kayuloweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weather(timezone: str, longitude: int, language: str, latitude: int, datasets: str='currentWeather,forecastDaily,forecastHourly,forecastNextHour', currentasof: str=None, hourlystart: str=None, hourlyend: str=None, countrycode: str='US', dailyend: str=None, dailystart: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain weather data for the specified location.
		
		The `dataSets` query parameter is used to obtain different data, like forecasts and current weather. Thus not all query parameters are needed for every data set. See the description of the query parameters to identify, which parameter is needed for which data set."
    timezone: The name of the timezone to use for rolling up weather forecasts into daily forecasts.
        longitude: The longitude of the desired location.
        language: The IETF BCP 47 language tag to use for localizing responses.
        latitude: The latitude of the desired location.
        datasets: A comma-delimited list of data sets to include in the response.
        currentasof: The time to obtain current conditions. Defaults to now. (This parameter is relevant for current weather only and cannot be used in combination with other data sets.)
Example: 2022-12-01T10:00:00Z
        hourlystart: The time to start the hourly forecast. If this parameter is absent, hourly forecasts start on the current hour. (This parameter is relevant for hourly forecasts only and cannot be used in combination with other data sets.)
Example: 2022-12-01T10:00:00Z
        hourlyend: The time to end the hourly forecast. If this parameter is absent, hourly forecasts run 24 hours or the length of the daily forecast, whichever is longer. (This parameter is relevant for hourly forecasts only and cannot be used in combination with other data sets.)
Example: 2022-12-01T10:00:00Z
        countrycode: The ISO Alpha-2 country code for the requested location. This parameter is necessary for weather alerts.
        dailyend: The time to end the daily forecast. If this parameter is absent, daily forecasts run for 10 days. (This parameter is relevant for daily forecasts onlyand cannot be used in combination with other data sets.)
Example: 2022-12-01T10:00:00Z
        dailystart: The time to start the daily forecast. If this parameter is absent, daily forecasts start on the current day. (This parameter is relevant for daily forecasts only and cannot be used in combination with other data sets.)
Example: 2022-12-01T10:00:00Z
        
    """
    url = f"https://kayuloweather.p.rapidapi.com/weather/{language}/{latitude}/{longitude}"
    querystring = {'timezone': timezone, }
    if datasets:
        querystring['dataSets'] = datasets
    if currentasof:
        querystring['currentAsOf'] = currentasof
    if hourlystart:
        querystring['hourlyStart'] = hourlystart
    if hourlyend:
        querystring['hourlyEnd'] = hourlyend
    if countrycode:
        querystring['countryCode'] = countrycode
    if dailyend:
        querystring['dailyEnd'] = dailyend
    if dailystart:
        querystring['dailyStart'] = dailystart
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kayuloweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def availability(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Determine the data sets available for the specified location."
    latitude: The latitude of the desired location.
        longitude: The longitude of the desired location.
        
    """
    url = f"https://kayuloweather.p.rapidapi.com/availability/{latitude}/{longitude}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kayuloweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

