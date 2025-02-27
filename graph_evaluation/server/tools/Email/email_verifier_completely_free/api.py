import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_verification(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using our Email Verification service is free with attribution. We require a link back to companyurlfinder.com on any page the service is used. Attribution must be legible and use at least a 12-point font. 
		
		<a href="https://companyurlfinder.com">provided by CUF Services</a>"
    
    """
    url = f"https://email-verifier-completely-free.p.rapidapi.com/email-verification/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-verifier-completely-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

