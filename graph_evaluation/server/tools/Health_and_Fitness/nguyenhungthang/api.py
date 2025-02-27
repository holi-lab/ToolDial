import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nguyenhungthang(nguyenhungthang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bác sĩ nam khoa"
    
    """
    url = f"https://nguyenhungthang.p.rapidapi.com/"
    querystring = {}
    if nguyenhungthang:
        querystring['nguyenhungthang'] = nguyenhungthang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nguyenhungthang.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

