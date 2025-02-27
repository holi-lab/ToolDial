import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_your_data(data: str, sleep: int=None, is_return: str=None, status: int=None, shuffle: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the data that you provide. The data parameter must be a valid JSON"
    
    """
    url = f"https://request-boomerang.p.rapidapi.com/query-json"
    querystring = {'data': data, }
    if sleep:
        querystring['sleep'] = sleep
    if is_return:
        querystring['return'] = is_return
    if status:
        querystring['status'] = status
    if shuffle:
        querystring['shuffle'] = shuffle
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "request-boomerang.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

