import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_verifier(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint verifies the validity of an email address. It checks the email address format, domain, and whether it is a disposable email address. It then extracts the MX records from the domain's DNS records and connects to the email server via SMTP. The function also simulates sending a message to the email server to confirm that the mailbox associated with the email address actually exists."
    
    """
    url = f"https://disposable-invalid-email-verifier.p.rapidapi.com/domain/email"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "disposable-invalid-email-verifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

