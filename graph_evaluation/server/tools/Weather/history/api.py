import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hourlyweather(lat: str, parameters: str, day: str, lng: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the historical weather on hourly level for a given day for a given location (latitude and longitude)"
    lat: The latitude in decimal format of the requested point
        parameters: Define the parameter, you wish to request. Allowed options are \"all\", \"air_quality\", \"anomaly\", \"astronomy\", \"weather\", \"signal\", \"pollen\"
        day: The requested day in the format \"YYYYmmdd\"
        lng: The longitude in decimal format of the requested point
        
    """
    url = f"https://history3.p.rapidapi.com/v0/hourly/{lng}/{lat}/{day}"
    querystring = {'parameters': parameters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "history3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hourlyweatherzip(parameters: str, zip_code: str, day: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the historical weather on hourly level for a given day for a given location (country and zip-code)"
    parameters: Define the parameter, you wish to request. Allowed options are \\\"all\\\", \\\"air_quality\\\", \\\"anomaly\\\", \\\"astronomy\\\", \\\"weather\\\", \\\"signal\\\", \\\"pollen\\\"
        zip_code: The postal code for the requested country
        day: The requested day in the format \\\"YYYYmmdd\\\"
        country: The ISO Alpha-2 code of the country
        
    """
    url = f"https://history3.p.rapidapi.com/v0/hourly-zip/{country}/{zip_code}/{day}"
    querystring = {'parameters': parameters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "history3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dailyweatherzip(country: str, zip_code: str, parameters: str, day: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the historical weather for a given day for a given location (Country and zip code)"
    country: The ISO Alpha-2 code of the country
        zip_code: The postal code for the requested country
        parameters: Define the parameter, you wish to request. Allowed options are \\\"all\\\", \\\"air_quality\\\", \\\"anomaly\\\", \\\"astronomy\\\", \\\"weather\\\", \\\"signal\\\", \\\"pollen\\\", \\\"occurrence\\\"
        day: The requested day in the format \\\"YYYYmmdd\\\"
        
    """
    url = f"https://history3.p.rapidapi.com/v0/daily-zip/{country}/{zip_code}/{day}"
    querystring = {'parameters': parameters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "history3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dailyweather(lng: str, lat: str, parameters: str, day: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the historical weather for a given day for a given location (latitude and longitude)"
    lng: The longitude in decimal format of the requested point
        lat: The latitude in decimal format of the requested point
        parameters: Define the parameter, you wish to request. Allowed options are \"all\", \"air_quality\", \"anomaly\", \"astronomy\", \"weather\", \"signal\", \"pollen\", \"occurrence\"
        day: The requested day in the format \"YYYYmmdd\"
        
    """
    url = f"https://history3.p.rapidapi.com/v0/daily/{lng}/{lat}/{day}"
    querystring = {'parameters': parameters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "history3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dailyweatherseries(lng: str, start_day: str, end_day: str, parameters: str, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the historical weather for a given series of days for a given location (latitude and longitude)"
    lng: The longitude in decimal format of the requested point
        start_day: The requested start day in the format \"YYYYmmdd\"
        end_day: The requested end day in the format \"YYYYmmdd\"
        parameters: Define the parameter, you wish to request. Allowed options are \"all\", \"air_quality\", \"anomaly\", \"astronomy\", \"weather\", \"signal\", \"pollen\", \"occurrence\"
        lat: The latitude in decimal format of the requested point
        
    """
    url = f"https://history3.p.rapidapi.com/v0/daily/{lng}/{lat}/{start_day}/{end_day}"
    querystring = {'parameters': parameters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "history3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

