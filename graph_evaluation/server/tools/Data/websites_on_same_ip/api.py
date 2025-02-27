import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_domains_websites_on_same_ip_shared(q: str, type: str, pagenum: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search domain and get other domains on same IP address, use IP address OR domain name"
    
    """
    url = f"https://websites-on-same-ip.p.rapidapi.com/"
    querystring = {'q': q, 'type': type, }
    if pagenum:
        querystring['pagenum'] = pagenum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "websites-on-same-ip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

