import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def google_hunt(api_key: str, email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find reviews and possible location of a google account."
    
    """
    url = f"https://checkleakedcc-official.p.rapidapi.com/ghunt"
    querystring = {'api-key': api_key, 'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checkleakedcc-official.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_lookup(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search ip location."
    
    """
    url = f"https://checkleakedcc-official.p.rapidapi.com/ip"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checkleakedcc-official.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leakcheck_net_query(check: str, type: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search in LeakCheck. You must add **192.126.161.211** to your authorized ips."
    check: Entry to be checked
        type: Check Type.
        key: LeakCheck.net API KEY (Not ours)
        
    """
    url = f"https://checkleakedcc-official.p.rapidapi.com/leak_check"
    querystring = {'check': check, 'type': type, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checkleakedcc-official.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

