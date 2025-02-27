import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getsuburbs(state: str='nsw', postcode: str=None, offset: int=None, name: str=None, limit: int=None, lga: str=None, abs_code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request and filter list of suburbs"
    state: Filter by state
        postcode: Filter by Post Code
        offset: How many items to skip
        name: Filter list by names
        limit: How many items to return
        lga: Filter by LGA
        abs_code: Filter by ABS SAL
        
    """
    url = f"https://australia-crime-api.p.rapidapi.com/suburb"
    querystring = {}
    if state:
        querystring['state'] = state
    if postcode:
        querystring['postcode'] = postcode
    if offset:
        querystring['offset'] = offset
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if lga:
        querystring['lga'] = lga
    if abs_code:
        querystring['abs_code'] = abs_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australia-crime-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

