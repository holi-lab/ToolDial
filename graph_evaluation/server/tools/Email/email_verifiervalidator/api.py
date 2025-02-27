import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## The status attribute will be:
		
		OK  - Only if the smtp server explicitly confirms the availability of the mailbox address.
		
		INVALID - If the smtp server explicitly confirms the mailbox is not available.
		
		UNKNOWN - For every other scenario (The mailbox may or may not be available)."
    
    """
    url = f"https://email-verifier-validator.p.rapidapi.com/"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-verifier-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

