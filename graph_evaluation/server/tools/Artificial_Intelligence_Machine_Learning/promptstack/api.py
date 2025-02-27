import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def prompt(user_id: str='<user_id>', prompt_id: str='<prompt_id>', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request to get prompt information with prompt_id"
    
    """
    url = f"https://promptstack.p.rapidapi.com/"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if prompt_id:
        querystring['prompt_id'] = prompt_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "promptstack.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group(group_id: str, user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request to get group information and list of all prompts of a given group with group_id"
    
    """
    url = f"https://promptstack.p.rapidapi.com/"
    querystring = {'group_id': group_id, 'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "promptstack.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

