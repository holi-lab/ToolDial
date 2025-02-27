import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def prematches(sort: str='league', date: str='31.12.2019', sportid: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bülten"
    
    """
    url = f"https://test1362.p.rapidapi.com/preMatches"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if date:
        querystring['date'] = date
    if sportid:
        querystring['sportId'] = sportid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test1362.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

