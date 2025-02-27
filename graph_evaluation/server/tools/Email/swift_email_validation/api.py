import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_checkemail_token_email(ip: str, email: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    ip: IP Address
        email: Email Address.
        token: Api Token.
        
    """
    url = f"https://swift-email-validation.p.rapidapi.com/{ip}/CheckEmail/{token}/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "swift-email-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_checkemail_token_tokeninfo(token: str, ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    token: Api Token.
        ip: IP Address
        
    """
    url = f"https://swift-email-validation.p.rapidapi.com/{ip}/CheckEmail/{token}/TokenInfo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "swift-email-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

