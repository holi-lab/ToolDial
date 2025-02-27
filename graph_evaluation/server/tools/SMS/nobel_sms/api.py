import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_sms_status(message_id: str, end_date: str, start_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get SMS status using Message ID"
    
    """
    url = f"https://nobel-sms.p.rapidapi.com/edr"
    querystring = {'message_id': message_id, 'end_date': end_date, 'start_date': start_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nobel-sms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

