import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_checker_api(email: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify emails in real-time"
    email: Email address to be verified
        key: The unique API key
        
    """
    url = f"https://emailchecker-email-verification-v1.p.rapidapi.com/api/a/v1?{key}=yourapikey&{email}=test@example.com"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emailchecker-email-verification-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

