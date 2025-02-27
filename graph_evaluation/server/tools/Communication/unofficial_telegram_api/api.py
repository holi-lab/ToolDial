import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_user_info(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user info from all accounts with customized handle. The API accepts both username(.e.g. dinonl) or user ID."
    
    """
    url = f"https://unofficial-telegram-api.p.rapidapi.com/getFullInfo"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-telegram-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_messages(is_id: str, from_id: int=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get messages in all public chats, groups and channel. Note: The user ID per message can be used through getFullInfo call to get user details."
    from_id: This will return all messages with ID that is bigger than the ID passed.
        limit: If no limit provided, default = 20
        
    """
    url = f"https://unofficial-telegram-api.p.rapidapi.com/getHistory/"
    querystring = {'id': is_id, }
    if from_id:
        querystring['from_id'] = from_id
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unofficial-telegram-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

