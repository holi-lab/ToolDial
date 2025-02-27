import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domain_or_email_address_check(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check a domain or email address if it is included in the wesbos list of known Disposable Email domains."
    query: The domain or email address to query.
        
    """
    url = f"https://disposable-email-address-detector-basic.p.rapidapi.com/api/dea/v1/check/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "disposable-email-address-detector-basic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

