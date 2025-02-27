import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reductsimpleurl(url: str, key: str, ndd: str=None, alias: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create shortened URL"
    url: Enter your url
        key: Key with 10 usage / day
        ndd: Chose you domain name (arl = arl.re or kurl = kurl.eu)
        alias: Chose a alias
        
    """
    url = f"https://reductible.p.rapidapi.com/"
    querystring = {'url': url, 'key': key, }
    if ndd:
        querystring['ndd'] = ndd
    if alias:
        querystring['alias'] = alias
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reductible.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

