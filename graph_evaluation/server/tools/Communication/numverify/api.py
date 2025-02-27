import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validate(number: str, country_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Main API endpoint used to validate phone numbers"
    number: the phone number to be validated
        country_code: Specify a country code if you intend to use a national phone number for your request
        
    """
    url = f"https://apilayer-numverify-v1.p.rapidapi.com/validate"
    querystring = {'number': number, }
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-numverify-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

