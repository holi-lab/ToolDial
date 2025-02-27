import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getphone(phone: str='2069735100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reverse Phone API validates a phone number and provides phone metadata, owner name and demographics, current and historical addresses, relatives and associated people, and additional phone numbers."
    
    """
    url = f"https://reverse-phone-api.p.rapidapi.com/3.1/phone"
    querystring = {}
    if phone:
        querystring['phone'] = phone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reverse-phone-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

