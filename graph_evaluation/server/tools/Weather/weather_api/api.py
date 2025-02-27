import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def country_weather(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Weather API provides developers with real-time weather data for locations worldwide. With this API, you can integrate weather information into your applications, websites, or services to deliver accurate forecasts and current weather conditions to your users. Accessible through easy-to-use endpoints, the Weather API offers various data points such as temperature, humidity, wind speed, and more."
    
    """
    url = f"https://weather-api92.p.rapidapi.com/weather"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weather-api92.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

