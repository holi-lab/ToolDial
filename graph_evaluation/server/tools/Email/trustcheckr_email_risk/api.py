import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trustcheckr_email_risk(email: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the email address and get  the risk associated with the email."
    
    """
    url = f"https://trustcheckr-email-risk.p.rapidapi.com/api/email/v1.3"
    querystring = {}
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trustcheckr-email-risk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

