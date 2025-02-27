import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def emailsherlock(email: str, client_ref: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "EmailSherlock _api"
    email: Email address to be searched
        client_ref: IP address of the client sending the request
        
    """
    url = f"https://emailsherlock.p.rapidapi.com/"
    querystring = {'email': email, 'client_ref': client_ref, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emailsherlock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

