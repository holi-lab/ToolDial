import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getapi(redirect_uri: str='code', response_type: str='https://www.instagram.com/oauth/authorize?', app_id: str='846745499103512&', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use Get Api"
    
    """
    url = f"https://myapp10.p.rapidapi.com/home/instragram"
    querystring = {}
    if redirect_uri:
        querystring['redirect_uri'] = redirect_uri
    if response_type:
        querystring['response_type'] = response_type
    if app_id:
        querystring['app_id'] = app_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "myapp10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

