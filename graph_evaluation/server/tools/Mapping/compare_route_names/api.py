import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def compare_route_names(str2: str, str1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculates the coefficient of how similar are 2 strings containing the name (and type) of the route. With a value of 0.9 and higher, it is possible to do auto-matching,at 0.2-0.9 - visual matching."
    
    """
    url = f"https://compare-route-names.p.rapidapi.com/"
    querystring = {'str2': str2, 'str1': str1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "compare-route-names.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

