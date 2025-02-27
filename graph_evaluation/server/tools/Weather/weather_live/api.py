import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_forecast_of_temperatures_for_the_next_14_days_for_a_particular_location(weatherid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the forecasted temperatures for the next 14 days in nearly any location around the world. (Index 0 is tomorrow, index 13 is 14 days from now)."
    
    """
    url = f"https://weather-live3.p.rapidapi.com/weather/forecast/{weatherid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-live3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_temperature_for_a_particular_location(weatherid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back the current temperature from nearly any location around the world."
    
    """
    url = f"https://weather-live3.p.rapidapi.com/weather/today/{weatherid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-live3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

