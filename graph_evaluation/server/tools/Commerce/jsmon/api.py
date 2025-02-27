import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_info(site_id: str, session_id: str, user_id: str, callback_id: str, action_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request information."
    site_id: Site ID
        session_id: Session ID
        user_id: User ID
        callback_id: Callback function
        action_type: Action ID
        
    """
    url = f"https://jsmon.p.rapidapi.com/track/"
    querystring = {'site_id': site_id, 'session_id': session_id, 'user_id': user_id, 'callback_id': callback_id, 'action_type': action_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jsmon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

