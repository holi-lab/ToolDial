import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_verification(emailaddress: str, outputformat: str='XML', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    emailaddress: email address which needs to be verified
        outputformat: XML | JSON (defaults to JSON)
        
    """
    url = f"https://whoisapi-email-verification-v1.p.rapidapi.com/api/v1"
    querystring = {'emailAddress': emailaddress, }
    if outputformat:
        querystring['outputformat'] = outputformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whoisapi-email-verification-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

