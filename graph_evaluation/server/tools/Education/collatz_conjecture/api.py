import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def statistics(start: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns statistics from the Collatz Conjecture function given a valid starting number."
    start: Any number less than or equal to 9223372036854775807.
        
    """
    url = f"https://collatz-conjecture.p.rapidapi.com/statistics"
    querystring = {'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collatz-conjecture.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def values(start: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of values and step statistics returned from the Collatz Conjecture function given a valid starting number."
    start: Any number less than or equal to 9223372036854775807.
        
    """
    url = f"https://collatz-conjecture.p.rapidapi.com/values"
    querystring = {'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collatz-conjecture.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

