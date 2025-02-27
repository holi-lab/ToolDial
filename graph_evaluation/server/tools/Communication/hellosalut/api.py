import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hellosalut(lang: str='en-us, ja, es', ip: str='2.222.222.222', mode: str='auto', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    lang: Language code (note, currently supported only cs, de, en-gb, en-us, es, fr, is, ja and sk)
        ip: An IPv4 IP address
        mode: Use built in IP and language detection
        
    """
    url = f"https://fourtonfish-hellosalut.p.rapidapi.com/"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if ip:
        querystring['ip'] = ip
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fourtonfish-hellosalut.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

