import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def phone_number_validation_by_ip(number: str, ip: str, localitylanguage: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validates phone number using IP address."
    
    """
    url = f"https://phone-number-validation-by-ip-address.p.rapidapi.com/data/phone-number-validate-by-ip"
    querystring = {'number': number, 'ip': ip, }
    if localitylanguage:
        querystring['localityLanguage'] = localitylanguage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-number-validation-by-ip-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

