import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_sms(provider: str, password: str, username: str, phone_number: str, is_from: str, sms: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send SMS"
    
    """
    url = f"https://sms136.p.rapidapi.com/send-sms"
    querystring = {'provider': provider, 'password': password, 'username': username, 'phone_number': phone_number, 'from': is_from, 'sms': sms, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms136.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

