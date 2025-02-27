import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cars_request(destination: int, dropoff: str, pickup: str, page: int=1, size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of available cars"
    destination: A destination to make an availability request
        dropoff: Dropoff date
        pickup: Pickup date
        page: Page number in a paginated list
        size: Page size in a paginated list (Default=10, Max = 20)
        
    """
    url = f"https://holiplus.p.rapidapi.com/cars/request.json"
    querystring = {'destination': destination, 'dropoff': dropoff, 'pickup': pickup, }
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holiplus.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

