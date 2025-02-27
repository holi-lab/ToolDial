import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domain_reputation(domainname: str, outputformat: str=None, mode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Evaluate reputation of a domain or IP (v1)"
    outputformat: Response output format. Acceptable values: XML or JSON. Defaults to JSON.
        mode: API can check your domain in 2 modes: 'fast' - some heavy tests and data collectors will be disabled. 'full' - all the data and the tests will be processed. Default: fast
        
    """
    url = f"https://domain-reputation1.p.rapidapi.com/api/v1"
    querystring = {'domainName': domainname, }
    if outputformat:
        querystring['outputFormat'] = outputformat
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-reputation1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

