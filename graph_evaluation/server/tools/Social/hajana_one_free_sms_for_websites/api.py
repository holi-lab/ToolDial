import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_sms(api: str='**********', number: str='03001234567', message: str='Test+SMS+here', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A File for sending SMS"
    api: Get Your Free API Key From Hajana One
        number: Pakistani Phone Number Only
        message: SMS Content Perameter
        
    """
    url = f"https://hajana1-hajana-one-free-sms-for-websites-v1.p.rapidapi.com/websms.php"
    querystring = {}
    if api:
        querystring['api'] = api
    if number:
        querystring['number'] = number
    if message:
        querystring['message'] = message
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hajana1-hajana-one-free-sms-for-websites-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

