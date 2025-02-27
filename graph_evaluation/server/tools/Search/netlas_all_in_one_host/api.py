import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def host_read(host: str, source_type: str='include', fields: str='["*"]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Both ip address and domain might be used as a search query."
    source_type: Include or exclude fields (choices: include, exclude)
        fields: Comma-separated list of fields to include/exclude
        
    """
    url = f"https://netlas-all-in-one-host.p.rapidapi.com/host/{host}/"
    querystring = {}
    if source_type:
        querystring['source_type'] = source_type
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netlas-all-in-one-host.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

