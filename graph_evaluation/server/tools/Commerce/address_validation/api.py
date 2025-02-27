import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validate_address(accept: str, country: str, street_1: str, street_number: str, postal_code: str, city: str='Oakville', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to validate address"
    
    """
    url = f"https://address-validation.p.rapidapi.com/api/rapi/addressvalidation/validate_address"
    querystring = {'Accept': accept, 'country': country, 'street_1': street_1, 'street_number': street_number, 'postal_code': postal_code, }
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

