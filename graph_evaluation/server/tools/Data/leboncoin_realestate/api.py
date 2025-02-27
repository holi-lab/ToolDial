import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def realestate_10000(realestate_10000: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "last 10000 realestate listings"
    
    """
    url = f"https://leboncoin_realestate.p.rapidapi.com/realestate_10000"
    querystring = {}
    if realestate_10000:
        querystring['realestate_10000'] = realestate_10000
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "leboncoin_realestate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

