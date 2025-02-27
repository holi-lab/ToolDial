import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_top_level_domains(tlds: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an exhaustive list of more than official 1500 generic and country code TLDs for which domains can be registered with."
    tlds: This queries to check if a specific TLD exist.  If the queried TLDs are found, they are returned in a Json Array. If none are found an empty json array [ ] is returned.
        
    """
    url = f"https://tld1.p.rapidapi.com/GetTlds"
    querystring = {}
    if tlds:
        querystring['tlds'] = tlds
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tld1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

