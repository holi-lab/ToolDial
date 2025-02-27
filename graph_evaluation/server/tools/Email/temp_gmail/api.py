import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get(domain: str, username: str, type: str='real', server: str='server-1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "server-1: 300 gmail real
		server-2: 1000 gmail real
		server-3:1000 gmail real"
    
    """
    url = f"https://temp-gmail.p.rapidapi.com/get"
    querystring = {'domain': domain, 'username': username, }
    if type:
        querystring['type'] = type
    if server:
        querystring['server'] = server
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temp-gmail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read_message(message_id: str, email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read Message"
    
    """
    url = f"https://temp-gmail.p.rapidapi.com/read"
    querystring = {'message_id': message_id, 'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temp-gmail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check(email: str, timestamp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check Inbox"
    
    """
    url = f"https://temp-gmail.p.rapidapi.com/check"
    querystring = {'email': email, 'timestamp': timestamp, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temp-gmail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

