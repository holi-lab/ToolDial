import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validate_email_address(email: str, catch_all: int=0, smtp: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The main API endpoint for validating email addresses"
    email: the email address to be validated
        catch_all: Set to "0" (default) to turn off Catch-all detection, set to "1" to turn on Catch-all detection
        smtp: Set to "0" to turn off SMTP check, set to "1" (default) to turn on SMTP check
        
    """
    url = f"https://apilayer-mailboxlayer-v1.p.rapidapi.com/check"
    querystring = {'email': email, }
    if catch_all:
        querystring['catch_all'] = catch_all
    if smtp:
        querystring['smtp'] = smtp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-mailboxlayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

