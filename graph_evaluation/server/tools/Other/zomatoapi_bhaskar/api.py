import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def singavaram_bhaskar(city: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://zomato62.p.rapidapi.com/restaurant"
    
    """
    url = f"https://zomatoapi-bhaskar.p.rapidapi.com/"
    querystring = {}
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zomatoapi-bhaskar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

