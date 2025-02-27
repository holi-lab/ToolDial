import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify_email(email: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks if email address exist and can be delivered.
		Checks the following for each email
		
		- email address valid
		- host is valid
		- host can accept the email RCPT
		- Is this CatchAll host"
    format: Response in JSON or XML format
        
    """
    url = f"https://email-delivery-checker.p.rapidapi.com/verify/v1"
    querystring = {'email': email, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-delivery-checker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

