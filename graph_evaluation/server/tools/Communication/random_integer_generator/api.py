import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_integer_generator(min: int=10, max: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates Random Integer"
    min: Min endpoint of interval. If not given, Integer Minimum Value is used (-2^31)
        max: Max endpoint of interval. If not given, Integer Maximum Value is used (2^31-1)
        
    """
    url = f"https://randomgen.p.rapidapi.com/api/random/"
    querystring = {}
    if min:
        querystring['min'] = min
    if max:
        querystring['max'] = max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "randomgen.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

