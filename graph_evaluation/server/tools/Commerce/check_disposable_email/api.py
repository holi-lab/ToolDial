import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def checkdisposableemail(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "REST API to lookup disposable email"
    email: Email to check disposable.
        
    """
    url = f"https://check-disposable-email.p.rapidapi.com/api/disposable"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-disposable-email.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailvalidation(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Clean your email list database with our free email checker and verifier"
    email: Email to be verified
        
    """
    url = f"https://check-disposable-email.p.rapidapi.com/api/email-validation"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-disposable-email.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def newlyregistereddomains(page: int=1, keywords: str=None, date: str='2023-06-12', exclude: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "REST API to lookup newly registered domains"
    page: page of domain to be searched
        keywords: contains keywords of domain to be searched
        date: registered date of domain to be searched
        exclude: exclude keywords of domain to be searched
        
    """
    url = f"https://check-disposable-email.p.rapidapi.com/api/newly-registered-domains"
    querystring = {}
    if page:
        querystring['page'] = page
    if keywords:
        querystring['keywords'] = keywords
    if date:
        querystring['date'] = date
    if exclude:
        querystring['exclude'] = exclude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-disposable-email.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def whois(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "REST API to WhoIS lookup data"
    domain: domain name of WhoIS to be searched
        
    """
    url = f"https://check-disposable-email.p.rapidapi.com/api/whois"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "check-disposable-email.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

