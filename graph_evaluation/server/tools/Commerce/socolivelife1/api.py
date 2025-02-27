import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def socolivelife(socolivelife: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://www.socolive.life/Socolive life là trang chính thức của Socolive TV"
    
    """
    url = f"https://socolivelife1.p.rapidapi.com/socolivelife"
    querystring = {}
    if socolivelife:
        querystring['socolivelife'] = socolivelife
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "socolivelife1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

