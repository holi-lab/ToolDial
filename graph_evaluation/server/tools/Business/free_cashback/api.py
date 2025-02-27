import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cashabck(ctype: str, limit: str, cid: str, sid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access this for cashback feed"
    
    """
    url = f"https://cashbackapi-free-cashback-v1.p.rapidapi.com/urecharge.in/coupons/cbapi.php?{ctype}=$ctype&{limit}=$limit&{cid}=$cid&{sid}=$sid"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cashbackapi-free-cashback-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

