import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_of_cities_in_india(deviceid: str, appversion: str, src: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of all cities in India
		Use city id to pull latest fuel price"
    
    """
    url = f"https://daily-fuel-prices-update-india.p.rapidapi.com/car/v2/fuel/cities"
    querystring = {'deviceId': deviceid, 'appVersion': appversion, 'src': src, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-fuel-prices-update-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_cities_in_india_state_wise(appversion: str, src: str, deviceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get state wise city list. Use the cityId in response to fetch fuel price for that particular city"
    
    """
    url = f"https://daily-fuel-prices-update-india.p.rapidapi.com/car/v2/fuel/states"
    querystring = {'appVersion': appversion, 'src': src, 'deviceId': deviceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-fuel-prices-update-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_fuel_price(cityid: str, appversion: str='1.0', deviceid: str='abcd', src: str='android-app', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest fuel price for the city provided as query param.
		Returns Petrol and Diesel Price of Current Day"
    
    """
    url = f"https://daily-fuel-prices-update-india.p.rapidapi.com/car/v2/fuel/prices"
    querystring = {'cityId': cityid, }
    if appversion:
        querystring['appVersion'] = appversion
    if deviceid:
        querystring['deviceId'] = deviceid
    if src:
        querystring['src'] = src
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-fuel-prices-update-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

