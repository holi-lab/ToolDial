import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def promotional_sms(sender: str, apikey: str, to: int, message: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Promotional sms Endpoint"
    
    """
    url = f"https://promotional-sms.p.rapidapi.com/"
    querystring = {'sender': sender, 'apikey': apikey, 'to': to, 'message': message, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "promotional-sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

