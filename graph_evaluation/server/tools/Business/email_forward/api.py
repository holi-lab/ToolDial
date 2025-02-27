import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def http_localhost_6000_smmmasters_api_v1_email_forwarder_key_abc_action_service(sfdsfs: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dzfdf"
    
    """
    url = f"https://email-forward.p.rapidapi.com/smmmasters/api/v1/email-forwarder?Key=abc&Action=service"
    querystring = {}
    if sfdsfs:
        querystring['sfdsfs'] = sfdsfs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-forward.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

