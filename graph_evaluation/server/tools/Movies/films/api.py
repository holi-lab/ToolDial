import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchinventory(searchstring: str='string', limit: int=0, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By passing in the appropriate options, you can search for
		available inventory in the system
		"
    searchstring: pass an optional search string for looking up inventory
        limit: maximum number of records to return
        skip: number of records to skip for pagination
        
    """
    url = f"https://films2.p.rapidapi.com/inventory"
    querystring = {}
    if searchstring:
        querystring['searchString'] = searchstring
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "films2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

