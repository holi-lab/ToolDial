import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def forecast_server_all_parameters(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch forecasts for a desired location for all parameters by specifying latitude, longitude, and desired timestamps."
    
    """
    url = f"https://excarta-apis.p.rapidapi.com/{latitude}/{longitude}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "excarta-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forecast_server(longitude: int, latitude: int, param: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch forecasts for a desired location by specifying latitude, longitude, parameter, and desired timestamps."
    
    """
    url = f"https://excarta-apis.p.rapidapi.com/{param}/{latitude}/{longitude}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "excarta-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

