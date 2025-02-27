import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def current_weather_report(lang: str, datatype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current Weather Report"
    
    """
    url = f"https://hourly-weather-report-of-hong-kong.p.rapidapi.com/weatherAPI/opendata/weather.php"
    querystring = {'lang': lang, 'dataType': datatype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hourly-weather-report-of-hong-kong.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

