import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_sms(to: str, message: str, connection_id: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send SMS"
    to: Destination Number
        message: Message
        connection_id: Connection ID
        password: Password
        
    """
    url = f"https://smslink.p.rapidapi.com/sendsms"
    querystring = {'to': to, 'message': message, 'connection_id': connection_id, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smslink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_balance(password: str, connection_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Account Balance"
    password: Password
        connection_id: Connection ID
        
    """
    url = f"https://smslink.p.rapidapi.com/credit"
    querystring = {'password': password, 'connection_id': connection_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smslink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

