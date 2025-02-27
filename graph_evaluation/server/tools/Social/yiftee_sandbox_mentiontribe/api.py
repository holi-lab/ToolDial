import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def balance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve account balance"
    
    """
    url = f"https://deepakrt123-yiftee-sandbox-mentiontribe.p.rapidapi.com/account/balance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "deepakrt123-yiftee-sandbox-mentiontribe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gift_status(audit_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    audit_number: string
        
    """
    url = f"https://deepakrt123-yiftee-sandbox-mentiontribe.p.rapidapi.com/gift/status"
    querystring = {'audit_number': audit_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "deepakrt123-yiftee-sandbox-mentiontribe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

