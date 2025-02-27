import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lirik(device: str, is_id: str, lang: str, regionuri: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Kotak7 lirik"
    
    """
    url = f"https://kotaklirik7.p.rapidapi.com/single"
    querystring = {'device': device, 'id': is_id, 'lang': lang, 'regionURI': regionuri, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kotaklirik7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

