import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_2022_12_gymshark_latestproducts(x_rapidapi_host: str=None, x_rapidapi_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest Products"
    
    """
    url = f"https://gymshark.p.rapidapi.com/2022-12/gymshark/latestproducts"
    querystring = {}
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gymshark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

