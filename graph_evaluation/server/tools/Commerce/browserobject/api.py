import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def browserobject_api(useragentstring: str, license: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Browser Detection API"
    useragentstring: The browser user agent string.
        license: API license key.
        
    """
    url = f"https://fraudlabs-browserobject-v1.p.rapidapi.com/browserobjectwebservice.asmx"
    querystring = {'USERAGENTSTRING': useragentstring, 'LICENSE': license, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fraudlabs-browserobject-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

