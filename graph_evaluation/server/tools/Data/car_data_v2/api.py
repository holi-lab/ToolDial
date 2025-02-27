import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cars(page: str, limit: str, type: str=None, model: str=None, make: str=None, year: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve and filter lists of cars"
    
    """
    url = f"https://car-data1.p.rapidapi.com/cars"
    querystring = {'page': page, 'limit': limit, }
    if type:
        querystring['type'] = type
    if model:
        querystring['model'] = model
    if make:
        querystring['make'] = make
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

