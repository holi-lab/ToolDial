import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def price(manufacturernumbase64encoded: str, condition: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest average price for manufacturer number"
    manufacturernumbase64encoded: A manufacturer number of an apple product, base64 encoded, e.g for a Apple MacBook Pro 13\\\" 2020 with manufacturer number MWP72D/A -> TVdQNzJEL0E=
        condition: ENUM, valid values are: USED, NEW, DEFECT, REFURBISHED

Note: Until now in v1, only values USED and REFURBISHED are allowed and give the same results (no distringuish between USED or REFURBISHED, so use any of the two.
In a future release, the other values will be also supported.
        
    """
    url = f"https://apfelpreise.p.rapidapi.com/price/{manufacturernumbase64encoded}"
    querystring = {'condition': condition, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apfelpreise.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

