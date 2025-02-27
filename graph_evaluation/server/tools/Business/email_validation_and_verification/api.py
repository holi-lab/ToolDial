import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_validation_api(email: str, auto_correct: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Email Validation and Verification API requires only a single email.
		Checking a misspelled email.
		Checking a malformed email."
    email: The email address to validate.
        auto_correct: You can chose to disable auto correct. To do so, just input  **false** for the auto correct param. By default, auto_correct is turned on.
        
    """
    url = f"https://email-validation-and-verification4.p.rapidapi.com/v1"
    querystring = {'email': email, }
    if auto_correct:
        querystring['auto_correct'] = auto_correct
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-validation-and-verification4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

