import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def current_weather_data_standard(coordinate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Global Forecast System (GFS) climate model is used to build the weather forecasts. With a spatial resolution of 0.25 degrees. The time resolution for the next 72 hours is 1 hour, and further, up to 384 hours - 3 hours. The GFS model is updated four times a day (00:00, 06:00, 12:00 and 18:00 UTC). All forecasts are provided by the UTC timestamp from the current timestamp for a period of up to 16 days.
		
		To obtain the weather forecast using this API, you must make a request for the URL using the GET method with the required parameters containing the required coordinates and the required timestamp. As a result, the response will be returned as a string in JSON format containing a list of heights and data related to these heights.
		
		**Parameters:**
		lat - Latitude, separator is a dot
		lon - Longitude, separator is a dot
		datetime - Time in the format YYYYmmddHHii. Where YYYY is the year, mm is the month, dd is the day, HH is the Hour, ii is the minute"
    
    """
    url = f"https://gridforecast.p.rapidapi.com/v1/forecast/{coordinate}/now"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gridforecast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_weather_data_full(coordinate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Global Forecast System (GFS) climate model is used to build the weather forecasts. With a spatial resolution of 0.25 degrees. The time resolution for the next 72 hours is 1 hour, and further, up to 384 hours - 3 hours. The GFS model is updated four times a day (00:00, 06:00, 12:00 and 18:00 UTC). All forecasts are provided by the UTC timestamp from the current timestamp for a period of up to 16 days.
		
		To obtain the weather forecast using this API, you must make a request for the URL using the GET method with the required parameters containing the required coordinates and the required timestamp. As a result, the response will be returned as a string in JSON format containing a list of heights and data related to these heights.
		
		**Parameters:**
		lat - Latitude, separator is a dot
		lon - Longitude, separator is a dot
		datetime - Time in the format YYYYmmddHHii. Where YYYY is the year, mm is the month, dd is the day, HH is the Hour, ii is the minute"
    
    """
    url = f"https://gridforecast.p.rapidapi.com/v1/forecast/full/{coordinate}/now"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gridforecast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

