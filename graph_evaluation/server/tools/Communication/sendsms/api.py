import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def message_send(password: str, to: int, text: str, action: str, username: str, is_from: str=None, report_mask: int=19, report_url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    password: Your SensSMS Api Key
        to: Destination Phone Number
        text: SMS Content
        username: Your SensSMS User Name
        is_from: Message Sender
        report_mask: Report Mask
        report_url: http://sendsms.ro/r_r.php?mySmsId=9876&status=%d
        
    """
    url = f"https://sendsms.p.rapidapi.com/json"
    querystring = {'password': password, 'to': to, 'text': text, 'action': action, 'username': username, }
    if is_from:
        querystring['from'] = is_from
    if report_mask:
        querystring['report_mask'] = report_mask
    if report_url:
        querystring['report_url'] = report_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sendsms.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

