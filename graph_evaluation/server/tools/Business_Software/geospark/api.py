import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def users(api_key: str, content_type: str, user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get users details"
    api_key: Api-Key
        content_type: Content-Type
        user_id: user_id
        
    """
    url = f"https://geospark.p.rapidapi.com/user/"
    querystring = {'Api-Key': api_key, 'Content-Type': content_type, 'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geospark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

