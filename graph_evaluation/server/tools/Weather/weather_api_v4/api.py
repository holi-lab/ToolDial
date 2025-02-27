import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weather_info_json(api_key: str='8af58654bb4cb8d7267be9b3a8e6759c', city: str='Paris', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end points takes  a 'GET' request with CITY as a parameter  and returns the real-time weather data for the CITY in a JSON format."
    
    """
    url = f"https://weather-api77.p.rapidapi.com/"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-api77.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

