import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_member(identifier: str, key: str='externalId', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a specific member. To do this, you need the `externalId` you gave the member when you created it or the `id` provided by Socie."
    
    """
    url = f"https://socie.p.rapidapi.com/api/v1/members/{identifier}"
    querystring = {}
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "socie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_members(limit: str='10', skip: str='20', sort: str='createdAt', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all or some members of your community."
    
    """
    url = f"https://socie.p.rapidapi.com/api/v1/members"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "socie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

