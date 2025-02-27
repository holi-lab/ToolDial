import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_airquality(x_rapidapi_key: str='2f918523acmsh6d5836c8338f131p1b2d83jsn251d1fe71e9c', x_rapidapi_host: str='air-quality-by-api-ninjas.p.rapidapi.com', city: str='Berlin', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://air-quality-demo-1.p.rapidapi.com/v1/airquality"
    querystring = {}
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "air-quality-demo-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

