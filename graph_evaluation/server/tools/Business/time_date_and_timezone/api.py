import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_time_endpoint(base_datetime: str, base_location: str, target_location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Convert Time Endpoint  makes it easy to convert the time and date of a given location to the time and date of another location. By default it converts the current time, but the conversion can take place in either the past or future with a simple parameter."
    base_datetime: The current datetime you're converting
        base_location: The location you use as a base to convert the datetime for. This parameter accepts the location as a string (e.g., Los Angeles, CA), a longitude and latitude (e.g., -31.4173391,-64.183319) , or an IP address (e.g., 82.111.111.111)
        target_location: The location you want to get the datetime for. This parameter accepts the location as a string (e.g., Los Angeles, CA), a longitude and latitude (e.g., -31.4173391,-64.183319) , or an IP address (e.g., 82.111.111.111)
        
    """
    url = f"https://time-date-and-timezone.p.rapidapi.com/v1/convert_time"
    querystring = {'base_datetime': base_datetime, 'base_location': base_location, 'target_location': target_location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "time-date-and-timezone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_time_endpoint(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Current Time Endpoint take a location in the form of a location name, latitude and longitude, or IP address and returns the current time, date, and timezone of that location."
    location: The location to determine the current time and timezone of. This parameter accepts the location as a string (e.g., Los Angeles, CA), a longitude and latitude (e.g., -31.4173391,-64.183319) , or an IP address (e.g., 82.111.111.111)
        
    """
    url = f"https://time-date-and-timezone.p.rapidapi.com/v1/current_time"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "time-date-and-timezone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

