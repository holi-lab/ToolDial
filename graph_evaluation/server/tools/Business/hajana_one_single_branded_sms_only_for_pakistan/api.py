import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def single_sms_api(phone: str, sender: str, username: str, password: str, message: str='This+is+my+test+SMS+from+Hajana-One+SMS+Gateway', uni: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API Send Single SMS"
    phone: Pakistani Phone Number
        sender: Sender ID Registered on Hajana One SMS Gateway
        username: Hajana One Account Username
        password: Hajana One Account Password
        message: Text Message Content
        uni: 1 for UniCode and 0 for English
        
    """
    url = f"https://hajana1-hajana-one-bulk-branded-sms-only-for-pakistan-v1.p.rapidapi.com/sms-api.php"
    querystring = {'phone': phone, 'sender': sender, 'username': username, 'password': password, }
    if message:
        querystring['message'] = message
    if uni:
        querystring['uni'] = uni
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hajana1-hajana-one-bulk-branded-sms-only-for-pakistan-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

