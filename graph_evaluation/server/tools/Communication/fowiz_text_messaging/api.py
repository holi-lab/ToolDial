import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sending_message(phonenumber: str, message: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sending message using FOWiz text messaging API"
    phonenumber: Recipient phone number
        message: Message to send
        
    """
    url = f"https://fowiz.p.rapidapi.com/message_http_api.php"
    querystring = {'phonenumber': phonenumber, 'message': message, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fowiz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

