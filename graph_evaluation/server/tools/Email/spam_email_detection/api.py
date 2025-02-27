import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def postvalues(tc0: str, pid: str, code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mandatory Parameters
		
		code=1SeE3KSQd4k0X7LePZDMYjUhRvfuPK8RpRYup6ZWe3RYAzFu5QxNtw==
		
		pid=cb52ef6b-179e-452f-a5de-5b189e7bcf89
		
		tc0=[message to test]"
    
    """
    url = f"https://spam-email-detection1.p.rapidapi.com/PostValues"
    querystring = {'tc0': tc0, 'pid': pid, 'code': code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spam-email-detection1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

