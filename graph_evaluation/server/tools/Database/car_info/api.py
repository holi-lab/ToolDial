import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_info(rego: str='fsd222', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Rego Info in New Zealand"
    
    """
    url = f"https://car-info.p.rapidapi.com/rego"
    querystring = {}
    if rego:
        querystring['rego'] = rego
    if rego:
        querystring['rego'] = rego
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

