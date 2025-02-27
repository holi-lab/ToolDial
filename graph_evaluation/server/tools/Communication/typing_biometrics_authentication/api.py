import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_user(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To check if a user exists and get the number of saved typing patterns. We recommend to save at least 2 typing patterns per user (170+ chars) in order to perform accurate authentications."
    id: A string of your choice that identifies the user
        
    """
    url = f"https://cristiandna-typing-biometrics-authentication-v1.p.rapidapi.com/user/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cristiandna-typing-biometrics-authentication-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

