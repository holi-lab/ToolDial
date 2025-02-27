import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getnumberreputation(is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method returns the reputation of a phone number."
    from: Telephone number in E.164 format ([+] [country code] [subscriber number including area code]).
        
    """
    url = f"https://robokiller-call-confidence.p.rapidapi.com/reputation"
    querystring = {'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "robokiller-call-confidence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

