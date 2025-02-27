import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mail(body: str, subject: str, sender: str, password: str, receiver: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the main endpoint where you have to set the query parameters as follows:
		sender: Your email address(with less secure apps enabled)
		password: Your email's password
		receiver: Email address of the recipient
		subject: Subject of the email
		body: body of the email"
    
    """
    url = f"https://e-mailer-420.p.rapidapi.com/mail/{sender}/{password}/{receiver}/{subject}/{body}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-mailer-420.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mail(body: str, subject: str, sender: str, password: str, receiver: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the main endpoint where you have to set the query parameters as follows:
		sender: Your email address(with less secure apps enabled)
		password: Your email's password
		receiver: Email address of the recipient
		subject: Subject of the email
		body: body of the email"
    
    """
    url = f"https://e-mailer-420.p.rapidapi.com/mail/{sender}/{password}/{receiver}/{subject}/{body}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "e-mailer-420.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

