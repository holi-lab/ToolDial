import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_validation(email: str, key: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Email Validation API does validation on a single email address and returns all the validation results in either JSON or XML format."
    
    """
    url = f"https://mailboxvalidator-com.p.rapidapi.com/validation/single"
    querystring = {'email': email, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mailboxvalidator-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def free_email(email: str, key: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Free Email API checks if a single email address is from a free email provider."
    
    """
    url = f"https://mailboxvalidator-com.p.rapidapi.com/email/free"
    querystring = {'email': email, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mailboxvalidator-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def disposable_email(key: str, email: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Disposable Email API helps users to detect and block disposable emails from sign up."
    
    """
    url = f"https://mailboxvalidator-com.p.rapidapi.com/email/disposable"
    querystring = {'key': key, 'email': email, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mailboxvalidator-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

