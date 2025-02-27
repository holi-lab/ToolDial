import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def phone_number_validation(localitylanguage: str='en', countrycode: str='us', number: str='201 867-5309', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API verifies phone number and helps your detect line type and format."
    
    """
    url = f"https://phone-number-validation.p.rapidapi.com/data/phone-number-validate"
    querystring = {}
    if localitylanguage:
        querystring['localityLanguage'] = localitylanguage
    if countrycode:
        querystring['countryCode'] = countrycode
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-number-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

