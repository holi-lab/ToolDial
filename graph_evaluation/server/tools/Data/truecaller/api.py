import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def phone_lookup(phone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a lot of valuable information about a specific number.
		
		**How to use**:
		{phone} - [country code][phone number].
		For example, for the number 0918217437 of Thailand, the correct value is 66918217437"
    
    """
    url = f"https://truecaller2.p.rapidapi.com/findPhone"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "truecaller2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

