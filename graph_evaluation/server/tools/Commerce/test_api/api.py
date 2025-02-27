import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sadasd(namesdf: str, name: str, sdfsdfsdf: str='sdfsdf', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "asdasdads"
    name: name
        sdfsdfsdf: sdfsdf
        
    """
    url = f"https://test-api25.p.rapidapi.com/testapi.com/name/{name}"
    querystring = {'Namesdf': namesdf, }
    if sdfsdfsdf:
        querystring['sdfsdfsdf'] = sdfsdfsdf
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test-api25.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

