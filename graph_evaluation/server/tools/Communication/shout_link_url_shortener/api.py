import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_url(is_id: str, password: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the original URL that was shortened"
    id: The ID that you want to retrieve
        password: The password used while shortening the URL
        
    """
    url = f"https://shoutlink.p.rapidapi.com/view/{is_id}"
    querystring = {}
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shoutlink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shorten_a_url(url: str, password: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a short memorable id (to be used while retrieving)"
    url: The original URL to be shortened
        password: If entered, the same password would be required while retrieving the URL
        
    """
    url = f"https://shoutlink.p.rapidapi.com/create"
    querystring = {'url': url, }
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shoutlink.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

