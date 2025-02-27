import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_my_bots(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Once you have created your bot, you can manage the configurations by using the “access_token” received during authentication. Here you can view a list of all the bots you have created on the platform."
    authorization: • Use access_token from Auth endpoint.
**example**: Bearer eyJhbGciOiJIUzI1NiJ9.NjZGU5YmYz.M73fb7oWW1ObQbmITdj1fxy0w--oGkx2iAVwFd_-5Us
        
    """
    url = f"https://aion.p.rapidapi.com/resources/bot/"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

