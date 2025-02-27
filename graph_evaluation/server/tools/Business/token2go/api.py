import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def read_a_link_or_token_by_its_id(is_id: str, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read a link or token by its id. It is used to see the content (t=link or t=token), but cannot validate the signature."
    
    """
    url = f"https://token2go.p.rapidapi.com/api/v1/token/{is_id}"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token2go.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_a_token(is_id: str, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Submit a token (t=token) or a link (t=link) id in order to check if  it is a valid one. To be a valid token it must have a valid signature and not expired."
    
    """
    url = f"https://token2go.p.rapidapi.com/api/v1/token/validate/{is_id}"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "token2go.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

