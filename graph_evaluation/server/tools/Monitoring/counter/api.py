import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_current_count(clabel: str='blue', color: str='red', idv: str='sample', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get count and and don't increase"
    
    """
    url = f"https://counter10.p.rapidapi.com/"
    querystring = {}
    if clabel:
        querystring['CLABEL'] = clabel
    if color:
        querystring['COLOR'] = color
    if idv:
        querystring['IDV'] = idv
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "counter10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_count_and_increase_by_one(is_id: str=None, clabel: str='blue', color: str='red', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get count and increase by one"
    
    """
    url = f"https://counter10.p.rapidapi.com/"
    querystring = {}
    if is_id:
        querystring['ID'] = is_id
    if clabel:
        querystring['CLABEL'] = clabel
    if color:
        querystring['COLOR'] = color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "counter10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

