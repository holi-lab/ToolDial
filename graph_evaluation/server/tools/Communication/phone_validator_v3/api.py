import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getphonebasic(phonenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a North American phone number (with or without a leading "1") and determine if it is a cell phone, landline, VOIP, toll-free or Unknown. Datapoints returned are linetype, phone company and location (City/State)."
    phonenumber: Any North American phone number, 10 or 11 digits (with or without leading "1"). Examples: 2025551234 or 12025551234.
        
    """
    url = f"https://phone-validator.p.rapidapi.com/api/v1/PhoneBasic/{phonenumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-validator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

