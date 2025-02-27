import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def send_mail_smtp(smtppass: str, smtpuser: str, recipient: str, smtpport: str, smtpurl: str, text: str=None, subject: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "send a mail from your own email provider (smtp)
		
		use the smtp-url (google -> *your provider* smtp) from your provider and your login (username and password)
		
		if your password contains reserved or unsafe characters -> encode the password to ASCII ( https://www.w3schools.com/tags/ref_urlencode.ASP )"
    
    """
    url = f"https://smtp-email.p.rapidapi.com/smtp/"
    querystring = {'smtpPass': smtppass, 'smtpUser': smtpuser, 'recipient': recipient, 'smtpPort': smtpport, 'smtpURL': smtpurl, }
    if text:
        querystring['text'] = text
    if subject:
        querystring['subject'] = subject
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "smtp-email.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

