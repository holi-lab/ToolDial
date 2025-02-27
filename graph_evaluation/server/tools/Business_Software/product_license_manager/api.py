import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def activate_license_ac(lic: str, key: str, func: str, api: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will activate the license which you generated using other endpoint, this license will be ready to use after activation.
		
		'func' stands for FUNCTION, it basically tells api what function user has requested to access. accessable functions are: gl,ac,st.
		gl: Generate license
		ac: Activate license
		st: Status
		
		'lic' stands for LICENSE, you need to givge an license which you want it to activate."
    
    """
    url = f"https://product-license-manager.p.rapidapi.com/webhook"
    querystring = {'lic': lic, 'key': key, 'func': func, 'api': api, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "product-license-manager.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def client_verification(key: str, api: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this you can add our license to any of you product. what id does is: It require an active license & key
		
		License is generatable & activationable through other endpoints.
		Key is basically a hardcodded value so we dont get unusual request.
		Once it get both things it will responde with 1 value either 0 or 1
		
		0: False
		1: True"
    
    """
    url = f"https://product-license-manager.p.rapidapi.com/client"
    querystring = {'key': key, 'api': api, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "product-license-manager.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

