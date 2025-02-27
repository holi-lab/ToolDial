import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def update(info3: str, info1: str, info2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "update endpoint"
    
    """
    url = f"https://ptl.p.rapidapi.com/update"
    querystring = {'info3': info3, 'info1': info1, 'info2': info2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ptl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_client(client_email: str, client_lastname: str, client_firstname: str, client_pass: str, client_login: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive new users"
    
    """
    url = f"https://ptl.p.rapidapi.com/create_client"
    querystring = {'client_email': client_email, 'client_lastname': client_lastname, 'client_firstname': client_firstname, 'client_pass': client_pass, 'client_login': client_login, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ptl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

