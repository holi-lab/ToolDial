import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def x_rapidapi_proxy_secret_123f7cd0_e73e_11e9_9851_17fbbfd737cf(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "X-RapidAPI-Proxy-Secret: 123f7cd0-e73e-11e9-9851-17fbbfd737cf"
    
    """
    url = f"https://test1125.p.rapidapi.comhttp://10.16.70.71/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test1125.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

