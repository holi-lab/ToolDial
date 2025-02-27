import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check(domain: str, extended: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes a domain name and returns SSL information.  Optionally, the extend property can be set for additional information."
    
    """
    url = f"https://ssl-snitch.p.rapidapi.com/v1/check"
    querystring = {'domain': domain, }
    if extended:
        querystring['extended'] = extended
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ssl-snitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

