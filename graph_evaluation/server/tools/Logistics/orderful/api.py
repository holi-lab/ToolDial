import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def transactions(orderful_api_key: str, is_id: str, content_type: str='application/json', accept: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Transaction by ID"
    id: Orderful Transaction ID
        
    """
    url = f"https://orderful.p.rapidapi.com/transactions/{is_id}"
    querystring = {'orderful-api-key': orderful_api_key, }
    if content_type:
        querystring['Content-Type'] = content_type
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "orderful.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

