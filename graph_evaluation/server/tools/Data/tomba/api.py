import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def domainstatus(email: str=None, domain: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns domain status if is webmail or disposable."
    
    """
    url = f"https://tomba.p.rapidapi.com/domain-status"
    querystring = {}
    if email:
        querystring['email'] = email
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def enrichment(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Enrichment API lets you look up person and company data based on an email, For example, you could retrieve a person’s name, location and social handles from an email"
    
    """
    url = f"https://tomba.p.rapidapi.com/enrich"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domainsearch(domain: str, department: str='pr', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this endpoint to show different browser icons to your users. The code argument receives the browser code as it appears in your user /account/sessions endpoint. Use width, height and quality arguments to change the output settings."
    domain: Domain name from which you want to find the email addresses. For example, \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"stripe.com\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
        department: Get only email addresses for people working in the selected department(s).
        page: Specifies the number of email addresses to skip. The default is 1.
        
    """
    url = f"https://tomba.p.rapidapi.com/domain-search"
    querystring = {'domain': domain, }
    if department:
        querystring['department'] = department
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linkedinfinder(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint generates or retrieves the most likely email address from a Linkedin URL."
    url: The URL of the Linkedin. For example, "https://www.linkedin.com/in/alex-maccaw-ab592978".
        
    """
    url = f"https://tomba.p.rapidapi.com/linkedin"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailverifier(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "verify the deliverability of an email address."
    email: The email address you want to verify.
        
    """
    url = f"https://tomba.p.rapidapi.com/email-verifier/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailfinder(domain: str, first_name: str, last_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generates or retrieves the most likely email address from a domain name, a first name and a last name."
    domain: Domain name from which you want to find the email addresses. For example, "stripe.com"
        first_name: The person's first name. It doesn't need to be in lowercase.
        last_name: The person's last name. It doesn't need to be in lowercase.
        
    """
    url = f"https://tomba.p.rapidapi.com/email-finder"
    querystring = {'domain': domain, 'first_name': first_name, 'last_name': last_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authorfinder(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint generates or retrieves the most likely email address from a blog post url."
    url: The URL of the article. For example, "https://clearbit.com/blog/company-name-to-domain-api".
        
    """
    url = f"https://tomba.p.rapidapi.com/author-finder"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emailcount(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Domain name from which you want to find the email addresses."
    domain: The email address you want to find sources.
        
    """
    url = f"https://tomba.p.rapidapi.com/email-count"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tomba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

