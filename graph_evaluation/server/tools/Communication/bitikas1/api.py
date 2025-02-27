import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lopp1(aaa: str=None, bbb: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "niisamalopp"
    
    """
    url = f"https://bitikas1.p.rapidapi.comhttp://www.brunn.ee"
    querystring = {}
    if aaa:
        querystring['aaa'] = aaa
    if bbb:
        querystring['bbb'] = bbb
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitikas1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

