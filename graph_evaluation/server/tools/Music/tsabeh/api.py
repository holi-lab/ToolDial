import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def test(name: str, url: str='https://tsabeh.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://tsabeh.com"
    name: اسم المحطة
        name: اسم المحطة
        
    """
    url = f"https://tsabeh.p.rapidapi.com/tsabeh.com"
    querystring = {'Name': name, }
    if name:
        querystring['Name'] = name
    if url:
        querystring['URL'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tsabeh.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

