import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bin_number(bin: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information specific to the specified BIN Number"
    bin: 6 digit BIN Number
        format: Determines the returned format. "xml" or "json"
        
    """
    url = f"https://findbinnumbers.p.rapidapi.com/"
    querystring = {'bin': bin, 'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "findbinnumbers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mod_10_check(format: str, card: str='5555555555554444', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Note: This verifies if a credit card number passes the MOD 10 algorithm however it doesn't guarantee that the card number is genuine"
    format: Determines the returned format. "xml" or "json"
        card: Credit card number you wish to validate via MOD 10
        
    """
    url = f"https://findbinnumbers.p.rapidapi.com/"
    querystring = {'format': format, }
    if card:
        querystring['card'] = card
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "findbinnumbers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

