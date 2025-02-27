import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def byteplant_email_address_validation(emailaddress: str, timeout: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify Email Address"
    emailaddress: email address to validate (string)
        timeout: timeout in seconds (default: 30s, min: 5s, max: 300s)
        
    """
    url = f"https://community-byteplant-email-address-online-verification.p.rapidapi.com/verify"
    querystring = {'EmailAddress': emailaddress, }
    if timeout:
        querystring['Timeout'] = timeout
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-byteplant-email-address-online-verification.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

