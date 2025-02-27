import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def carbon_dioxide_endpoint(co2: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response will be an object with no CORS resources enabled."
    
    """
    url = f"https://daily-atmosphere-carbon-dioxide-concentration.p.rapidapi.com/api/co2-api"
    querystring = {}
    if co2:
        querystring['co2'] = co2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-atmosphere-carbon-dioxide-concentration.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

