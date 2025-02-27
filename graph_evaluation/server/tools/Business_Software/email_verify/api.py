import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_email(domain: str, ln: str, fn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It will find the email address of any Person by using the Name & Company Website. Get genuine Emails that are 100% verified with details like disposable, deliverable, full-inbox & catch-all."
    
    """
    url = f"https://email-verify6.p.rapidapi.com/single-find"
    querystring = {'domain': domain, 'ln': ln, 'fn': fn, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-verify6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_email(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify Email will check for the following: If the email is working and deliverable If the email address exists If the email is a disposable email If the email inbox is full or not If the email has a correct syntax"
    
    """
    url = f"https://email-verify6.p.rapidapi.com/single-verify"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-verify6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

