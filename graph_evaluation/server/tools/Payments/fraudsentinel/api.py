import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fraudsentinel(ip: str, nocache: bool=None, paranoid: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detailed JSON data from hundreds of IP classification databases."
    
    """
    url = f"https://fraudsentinel.p.rapidapi.com/sentinel.json"
    querystring = {'ip': ip, }
    if nocache:
        querystring['nocache'] = nocache
    if paranoid:
        querystring['paranoid'] = paranoid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fraudsentinel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

