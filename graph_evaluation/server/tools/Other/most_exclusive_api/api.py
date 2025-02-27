import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exclusiveness_callerid(callerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check Exclusiveness. See if you are already exclusive, not yet or you get the exclusivity now."
    callerid: Caller ID. Any string identifying the requester. Be creative. Alphanumeric [a-zA-Z0-9_-] max 32 chars.
        
    """
    url = f"https://most-exclusive-api.p.rapidapi.com/exclusiveness/{callerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "most-exclusive-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

