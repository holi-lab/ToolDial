import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domain_name_search(ip_assigned: str, q: str, type: str, pagenum: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search 600m+ root domains (domain names) by partial match"
    
    """
    url = f"https://domain-name-search4.p.rapidapi.com/"
    querystring = {'ip_assigned': ip_assigned, 'q': q, 'type': type, 'pagenum': pagenum, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-name-search4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

