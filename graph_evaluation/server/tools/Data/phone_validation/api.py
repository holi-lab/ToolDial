import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def phone(phone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info on phone number"
    phone: Provided number should be in E.164 format. (E.164 numbers can have a maximum of fifteen digits and are usually written as follows: [+][country code][subscriber number including area code].)
        
    """
    url = f"https://phone-validation.p.rapidapi.com/{phone}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

