import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getweatherforecast(location: str, days: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides weather forecast information based on the user's location. It returns the current weather conditions, daily forecasts, and additional weather data, such as temperature, humidity, and wind speed. The data is sourced from OpenWeatherMap, ensuring accurate and reliable information."
    location: The name of the city or location for which to retrieve the weather forecast.
        days: The number of days of forecast data to retrieve (1-16). If not provided, it defaults to 1 day.
        
    """
    url = f"https://apjoy-weather-forecast.p.rapidapi.com/forecast"
    querystring = {'location': location, }
    if days:
        querystring['days'] = days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apjoy-weather-forecast.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

