import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mailboxvalidator_free_email_checker_api(email: str, key: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Free Email API checks if a single email address is from a free email provider and returns the results in either JSON or XML format."
    email: The email address to check if is from a free email provider.
        key: Get started with 300 monthly credits for FREE at https://www.mailboxvalidator.com/plans#api
        format: Return the result in json (default) or xml format. Valid values: json | xml
        
    """
    url = f"https://mailboxvalidator-mailboxvalidator-single-validation-v1.p.rapidapi.com/v1/email/free"
    querystring = {'email': email, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mailboxvalidator-mailboxvalidator-single-validation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mailboxvalidator_single_validation(email: str, key: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Single Validation API does validation on a single email address and returns all the validation results in either JSON or XML format."
    email: The email address to validate
        key: Get started with 300 monthly credits for FREE at https://www.mailboxvalidator.com/plans#api
        format: Return format: json | xml
        
    """
    url = f"https://mailboxvalidator-mailboxvalidator-single-validation-v1.p.rapidapi.com/v1/validation/single"
    querystring = {'email': email, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mailboxvalidator-mailboxvalidator-single-validation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mailboxvalidator_disposable_email_checker_api(email: str, key: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Disposable Email API checks if a single email address is from a disposable email provider and returns the results in either JSON or XML format."
    email: The email address to check if is from a disposable email provider.
        key: Get started with 300 monthly credits for FREE at https://www.mailboxvalidator.com/plans#api
        format: Return the result in json (default) or xml format. Valid values: json | xml
        
    """
    url = f"https://mailboxvalidator-mailboxvalidator-single-validation-v1.p.rapidapi.com/v1/email/disposable"
    querystring = {'email': email, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mailboxvalidator-mailboxvalidator-single-validation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

