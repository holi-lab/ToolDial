import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def validate_single_email(email: str, apikey: str='rapid_solo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate single email"
    email: Email to check
        
    """
    url = f"https://myemailverifier1.p.rapidapi.com/validate_rapid"
    querystring = {'email': email, }
    if apikey:
        querystring['apikey'] = apikey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myemailverifier1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

