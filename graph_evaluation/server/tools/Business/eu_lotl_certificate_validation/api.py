import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def service_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check service status"
    
    """
    url = f"https://trust1team-eu-lotl-certificate-validation-v1.p.rapidapi.com/system/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trust1team-eu-lotl-certificate-validation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_challenge(digest: str='SHA256', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generates a new challenge. The digest algorithm can be: 'MD5', 'SHA1', 'SHA256' or 'SHA512'"
    digest: The digest algorithm can be: 'MD5', 'SHA1', 'SHA256' or 'SHA512'
        
    """
    url = f"https://trust1team-eu-lotl-certificate-validation-v1.p.rapidapi.com/challenge"
    querystring = {}
    if digest:
        querystring['digest'] = digest
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trust1team-eu-lotl-certificate-validation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

