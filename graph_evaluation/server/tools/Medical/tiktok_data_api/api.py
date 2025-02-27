import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def followers(max_time: str='0', type: str='1', uid: str='7098251395363390469', count: str='50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get tiktok followers"
    
    """
    url = f"https://tiktok-data-api.p.rapidapi.com/"
    querystring = {}
    if max_time:
        querystring['max_time'] = max_time
    if type:
        querystring['type'] = type
    if uid:
        querystring['uid'] = uid
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

