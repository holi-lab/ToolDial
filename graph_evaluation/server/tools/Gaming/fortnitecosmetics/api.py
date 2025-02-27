import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchinventory(skip: int=None, limit: int=None, searchstring: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By passing in the appropriate options, you can search for
		available inventory in the system
		"
    skip: number of records to skip for pagination
        limit: maximum number of records to return
        searchstring: pass an optional search string for looking up inventory
        
    """
    url = f"https://fortnitecosmetics.p.rapidapi.com/skins"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if searchstring:
        querystring['searchString'] = searchstring
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fortnitecosmetics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

