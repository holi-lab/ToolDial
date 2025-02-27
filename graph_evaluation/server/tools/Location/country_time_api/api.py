import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def time_api(country: str, x_rapidapi_host: str='country-time-api.p.rapidapi.com', x_rapidapi_key: str='2bfebb13b8msh9ea690772d3c966p14c54fjsn4cff9e431a76', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Time API is a simple and convenient web service that allows you to retrieve the current time for different countries around the world. By providing the country name as a parameter, you can obtain the accurate and up-to-date local time for that specific location."
    
    """
    url = f"https://country-time-api.p.rapidapi.com/time"
    querystring = {'country': country, }
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "country-time-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

