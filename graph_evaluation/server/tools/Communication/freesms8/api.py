import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_sms(uid: str, pwd: str, phone: str, msg: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send SMS using this endpoint."
    uid: Username registered on freesms8
        pwd: Password of the account registered on freesms8
        phone: Recipients 10-digit phone numbers separated by semicolon
        msg: Message to be sent as SMS
        
    """
    url = f"https://freesms8.p.rapidapi.com/index.php"
    querystring = {'uid': uid, 'pwd': pwd, 'phone': phone, 'msg': msg, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "freesms8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

