import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def normalize(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get normalized form of an email"
    email: email id
        
    """
    url = f"https://email-utilities.p.rapidapi.com/normalize"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-utilities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify(email: str, checksmtp: bool=None, suggestdomain: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get normalized form of an email"
    email: email id
        checksmtp: Is SMTP check required on port 25
        suggestdomain: Is Domain suggestion needed
        
    """
    url = f"https://email-utilities.p.rapidapi.com/verify"
    querystring = {'email': email, }
    if checksmtp:
        querystring['checkSMTP'] = checksmtp
    if suggestdomain:
        querystring['suggestDomain'] = suggestdomain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-utilities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

