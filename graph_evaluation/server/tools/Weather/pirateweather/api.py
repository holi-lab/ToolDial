import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pirateweather_forecast(location: str, extend: str=None, units: str=None, exclude: str=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the primary API endpoint for the PirateWeather Forecast. A detailed description is available in the docs, but in short, this API takes NOAA weather forecasts, process them in a clear and documented way,  and then share them using a syntax compatible with the Dark Sky API. This provides a service to keep legacy applications running after the Dark Sky API shutdown date, as well as providing a weather data source that emphases data transparency!"
    
    """
    url = f"https://pirateweather.p.rapidapi.com/proxy/forecast/{location}"
    querystring = {}
    if extend:
        querystring['extend'] = extend
    if units:
        querystring['units'] = units
    if exclude:
        querystring['exclude'] = exclude
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pirateweather.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

