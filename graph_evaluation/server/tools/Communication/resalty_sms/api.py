import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_your_account_balance(password: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://mob4me.p.rapidapi.com/getBalance.php"
    querystring = {'password': password, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mob4me.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sendsms_php(userid: str, password: str, sender: str, to: int, msg: str='test me', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    sender: phone number registered on your profile
        to: add phone Number, no need to add (+) or (00) in country code
        
    """
    url = f"https://mob4me.p.rapidapi.com/sendSMS.php"
    querystring = {'userid': userid, 'password': password, 'sender': sender, 'to': to, }
    if msg:
        querystring['msg'] = msg
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mob4me.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

