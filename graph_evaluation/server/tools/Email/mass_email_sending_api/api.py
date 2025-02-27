import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_mass_email_endpoint(mail_password: str, mail_port: int, subject: str, body: str, mail_username: str, sender: str, file_path: str, mail_server: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/send_email"
    
    """
    url = f"https://mass-email-sending-api.p.rapidapi.com/send_email"
    querystring = {'mail_password': mail_password, 'mail_port': mail_port, 'subject': subject, 'body': body, 'mail_username': mail_username, 'sender': sender, 'file_path': file_path, 'mail_server': mail_server, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mass-email-sending-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

