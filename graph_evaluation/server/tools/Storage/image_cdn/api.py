import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def show_history(pagesize: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows all images you uploaded in a pageable list"
    pagesize: The size of one page, maximum 100
        page: The page you want to return, starting at 0
        
    """
    url = f"https://image-cdn.p.rapidapi.com/list"
    querystring = {'pageSize': pagesize, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-cdn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

