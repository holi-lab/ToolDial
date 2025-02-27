import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def format_number_for_region(number: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Formats a given phone number for a specified region"
    number: Provide a complete phone number
        country: Provide a two letter country code
        
    """
    url = f"https://phone-number-validator3.p.rapidapi.com/format"
    querystring = {'number': number, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-number-validator3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_for_region(number: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validates phone number for a specified region"
    number: Provide a complete phone number
        country: Provide a two letter country code
        
    """
    url = f"https://phone-number-validator3.p.rapidapi.com/validate"
    querystring = {'number': number, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-number-validator3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

