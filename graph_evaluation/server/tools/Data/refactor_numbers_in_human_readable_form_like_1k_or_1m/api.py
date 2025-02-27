import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def number(number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a number to get the refactored human-readable form of the number.
		You can use this to show the price for your products to your e-commerce website, and many more."
    
    """
    url = f"https://refactor-numbers-in-human-readable-form-like-1k-or-1m.p.rapidapi.com/{number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "refactor-numbers-in-human-readable-form-like-1k-or-1m.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

