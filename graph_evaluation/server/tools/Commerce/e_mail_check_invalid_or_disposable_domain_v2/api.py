import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mailcheck(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if e-mail domain is valid, or a disposable/temporary address. Invalid domains (typos, non-responding mailserver, etc) will return "valid: false", "block: true". Disposable e-mail domains will return "valid: true" (since it's a valid domain), but "block: true" and "disposable: true"."
    domain: Full e-mail, or domain to check if valid or temporary/disposable. You can enter an e-mail address, and it will be converted to a domain, but entering just the domain is recommended for user privacy reasons.
        
    """
    url = f"https://mailcheck.p.rapidapi.com/"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mailcheck.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

