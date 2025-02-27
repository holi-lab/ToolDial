import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def role_account(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the rols account of your domain"
    
    """
    url = f"https://fast-email-verifier.p.rapidapi.com/IsRoleAccount"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-email-verifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def freedomain(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check whether or not you have a free domain."
    
    """
    url = f"https://fast-email-verifier.p.rapidapi.com/FreeDomain"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-email-verifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def disposable_domain_and_suggest_domain(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find out if the domain is disposable and get a suggested domain."
    
    """
    url = f"https://fast-email-verifier.p.rapidapi.com/disposableDomain"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-email-verifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def email_check_smtp(username: str, domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An easy way of checking your domain's SMTP."
    
    """
    url = f"https://fast-email-verifier.p.rapidapi.com/emailCheckSMTP"
    querystring = {'username': username, 'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-email-verifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailverifications(emailsimple: str='mbavazijoshua@gmail.com,apisolution@gmail.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "He is  a simple way  to verify a list of email addresses with different dimensions."
    
    """
    url = f"https://fast-email-verifier.p.rapidapi.com/EmailVerifications"
    querystring = {}
    if emailsimple:
        querystring['emailsimple'] = emailsimple
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-email-verifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailverification(emailsimple: str='mbavazijoshua@gmail.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Basic usage
		It is a simple way to verify an email address with different dimensions."
    
    """
    url = f"https://fast-email-verifier.p.rapidapi.com/EmailVerification"
    querystring = {}
    if emailsimple:
        querystring['emailsimple'] = emailsimple
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fast-email-verifier.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

