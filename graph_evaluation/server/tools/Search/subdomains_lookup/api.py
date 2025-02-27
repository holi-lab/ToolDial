import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def subdomains_lookup(type: str, pagenum: int, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query API with root domain to retrieve all known subdomains"
    
    """
    url = f"https://subdomains-lookup1.p.rapidapi.com/"
    querystring = {'type': type, 'pagenum': pagenum, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "subdomains-lookup1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

