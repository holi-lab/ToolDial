import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_validatephone(number: str, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Validate Phone API endpoint. Returns metadata (including whether it is valid) for a given phone number."
    number: phone number to check. If country is not set, the 3-digit country code prefix needs to be included.
        country: 2-letter ISO-3166 country code the phone number belongs to.
        
    """
    url = f"https://validate-phone-by-api-ninjas.p.rapidapi.com/v1/validatephone"
    querystring = {'number': number, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "validate-phone-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

