import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def testendpoint(is_id: int, filters: str='["sort_alphabetical"]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "dsafasdfdfdsafdsafdsafdsaf"
    
    """
    url = f"https://test4935.p.rapidapi.com/purchases/{is_id}"
    querystring = {}
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test4935.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

