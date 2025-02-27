import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/v3/verify endpoint takes an email address as input, does a full verification, and tells you if emails are deliverable to this address or not. Verimail's full verification include:
		
		- Syntax verification
		- Domain's MX records verifications
		- Full SMTP verification on the Email Service Provider's servers
		- Disposable or temporary address detection
		- Catch-all domains detection
		
		It will also detect typos for most free email providers (like gmail, yahoo, outlook) and suggest a correction."
    email: the email address to be verified
        
    """
    url = f"https://verimail.p.rapidapi.com/v3/verify"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "verimail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

