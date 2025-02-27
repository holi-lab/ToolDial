import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_address(email: str, ext: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test deliverability of an email address and see if it bounces or not."
    email: Email address to be verified
        ext: (Optional) Extended information about the e-mail address (0 or 1)
        
    """
    url = f"https://email-address-verification.p.rapidapi.com/email/api.php"
    querystring = {'email': email, }
    if ext:
        querystring['ext'] = ext
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-address-verification.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

