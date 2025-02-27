import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_urllookup(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas URL Lookup API endpoint."
    url:  valid URL to check. It supports schemes (e.g. http://example.com) as well as schemeless (e.g. example.com) formats.
        
    """
    url = f"https://url-lookup-by-api-ninjas.p.rapidapi.com/v1/urllookup"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "url-lookup-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

