import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_smartphone(phone: str, brand: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get a specific smartphone, you first define your brand, followed by a slash, followed by you specific phone.
		
		spaces in smartphone-names are replaced with "_""
    
    """
    url = f"https://smartphone-specifications.p.rapidapi.com/{brand}/{phone}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smartphone-specifications.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

