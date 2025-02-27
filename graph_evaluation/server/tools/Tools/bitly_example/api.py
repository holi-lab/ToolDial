import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sendsms(msg: str=None, phone: str=None, pwd: str=None, uid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://rashmi_tembhurne-bitly-example-v1.p.rapidapi.com/v3/shorten"
    querystring = {}
    if msg:
        querystring['msg'] = msg
    if phone:
        querystring['phone'] = phone
    if pwd:
        querystring['pwd'] = pwd
    if uid:
        querystring['uid'] = uid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rashmi_tembhurne-bitly-example-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

