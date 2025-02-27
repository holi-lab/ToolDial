import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def items(is_id: int=1, name: str='CHAOS FLAME EMBER', type: str='Ember', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Itens informations"
    
    """
    url = f"https://souls-api.p.rapidapi.com/items/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "souls-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

