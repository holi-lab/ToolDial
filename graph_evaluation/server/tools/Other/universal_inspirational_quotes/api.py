import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_topics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delivers available Inspirational Topics"
    
    """
    url = f"https://healthruwords.p.rapidapi.com/v1/topics/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "healthruwords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_quotes(t: str='Wisdom', maxr: int=1, size: str='medium', is_id: int=731, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delivers Inspirational Quotes - if used without parameters, it will return one random quote (default behavior)"
    t: List of topics within which selection one or several quotes will be returned
        maxr: Max number of quotes returned - Default: 1
        size: Defines the size of the image to be returned. Possible values: medium, thumbnail. Default value: thumbnail
        is_id: Id of the quote - if specified a random quote will not be returned.
        
    """
    url = f"https://healthruwords.p.rapidapi.com/v1/quotes/"
    querystring = {}
    if t:
        querystring['t'] = t
    if maxr:
        querystring['maxR'] = maxr
    if size:
        querystring['size'] = size
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "healthruwords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

