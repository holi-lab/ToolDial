import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def electricity_price_daily_average(zone: str, timestamp: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns JSON with price average for the day that matches the timestamp in the given zone.
		Returned "time" object is  l o c a l time in the given zone.
		   
		Parameter 1:
		number - timestamp: in s e c o n d s unix time U T C
		if timestamp is 0, current local time for the zone will be used
		   
		Parameter 2:
		String - zone: either peninsular, canarias, baleares, ceuta or melilla"
    
    """
    url = f"https://pvpc-hourly-spanish-energy-prices-api.p.rapidapi.com/price-average/{timestamp}/{zone}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pvpc-hourly-spanish-energy-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def electricity_price_daily(timestamp: str, zone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns JSON with price data for the day that matches the timestamp in the given zone.
		Returned "time" object is  l o c a l time in the given zone.
		
		Parameter 1:
		number - timestamp: in s e c o n d s unix time U T C
		if timestamp is 0, current day for the zone will be used
		
		Parameter 2:
		String - zone: either peninsular, canarias, baleares, ceuta or melilla"
    
    """
    url = f"https://pvpc-hourly-spanish-energy-prices-api.p.rapidapi.com/price-daily/{timestamp}/{zone}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pvpc-hourly-spanish-energy-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def electricity_price_hourly(zone: str, timestamp: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns JSON with price data for the hour that matches the timestamp in the given zone.
		Returned "time" object is  l o c a l time in the given zone.
		   
		Parameter 1:
		number - timestamp: in s e c o n d s unix time U T C
		if timestamp is 0, current local time for the zone will be used
		   
		Parameter 2:
		String - zone: either peninsular, canarias, baleares, ceuta or melilla"
    
    """
    url = f"https://pvpc-hourly-spanish-energy-prices-api.p.rapidapi.com/price/{timestamp}/{zone}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pvpc-hourly-spanish-energy-prices-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

