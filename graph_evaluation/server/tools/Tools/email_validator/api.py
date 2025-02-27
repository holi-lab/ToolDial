import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_validator_validate(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate email address. Key features:
		1. Checks that an email address has the correct syntax
		1. Gives friendly error messages when validation fails.
		1. Checks deliverability of an email address.
		1. Supports internationalized domain names and 
		1. Normalizes email addresses"
    
    """
    url = f"https://email-validator28.p.rapidapi.com/email-validator/validate"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-validator28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def email_validator_health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the health status of the API. Returns current UTC time."
    
    """
    url = f"https://email-validator28.p.rapidapi.com/email-validator/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-validator28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

