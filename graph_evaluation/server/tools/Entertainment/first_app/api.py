import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def test(param2: str='tesst2', param1: str='test', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "test thoi"
    param2: this ist test2
        param1: this ist test1
        
    """
    url = f"https://first-app.p.rapidapi.com/test"
    querystring = {}
    if param2:
        querystring['param2'] = param2
    if param1:
        querystring['param1'] = param1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "first-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

