import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mailboxvalidator_api(email: str, license: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Email validation API"
    email: Email address to validate.
        license: API license key.
        
    """
    url = f"https://fraudlabs-mailboxvalidator-v1.p.rapidapi.com/mailboxvalidatorwebservice.asmx"
    querystring = {'EMAIL': email, 'LICENSE': license, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fraudlabs-mailboxvalidator-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

