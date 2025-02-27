import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send(username: str, password: str, message: str, group: str=None, number: str='(xxx)yyy-zzzz', send_at: str='2021-12-31 11:59:59', contact: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "use this to send the sms."
    
    """
    url = f"https://sms151.p.rapidapi.com/send/index.php"
    querystring = {'username': username, 'password': password, 'message': message, }
    if group:
        querystring['group'] = group
    if number:
        querystring['number'] = number
    if send_at:
        querystring['send_at'] = send_at
    if contact:
        querystring['contact'] = contact
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sms151.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

