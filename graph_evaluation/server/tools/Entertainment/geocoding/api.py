import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def broadcast_weather_forecast(continent: str='EU', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    continent: Webcams from Europe
        
    """
    url = f"https://produkce-geocoding-v1.p.rapidapi.com/broadcast/weather-forecast"
    querystring = {}
    if continent:
        querystring['continent'] = continent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "produkce-geocoding-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

