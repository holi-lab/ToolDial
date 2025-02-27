import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def new_call(new_get: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Method Calls a New Function"
    new_get: It Will Install the API on Server
        new_get: It Will Call the String
        
    """
    url = f"https://learn-online.p.rapidapi.com/Install_Header"
    querystring = {'NEW_Get': new_get, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "learn-online.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

