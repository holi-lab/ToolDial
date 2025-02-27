import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def carrier_endpoint(phonenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The phone number carrier provides useful information about location. Please any Bug contact me at @ta9ra9pa9 on Telegram"
    
    """
    url = f"https://free-phone-carrier-lookup.p.rapidapi.com/num/{phonenumber}"
    querystring = {}
    if phonenumber:
        querystring['PhoneNumber'] = phonenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-phone-carrier-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

