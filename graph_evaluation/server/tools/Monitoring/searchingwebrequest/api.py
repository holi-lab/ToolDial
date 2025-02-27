import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchwebrequest(keyword: str, response: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We need to get the response from the web after specifying a keyword"
    
    """
    url = f"https://searchingwebrequest.p.rapidapi.comhttp://localhost:54192/Panel/Mentions"
    querystring = {'Keyword': keyword, }
    if response:
        querystring['Response'] = response
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "searchingwebrequest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

