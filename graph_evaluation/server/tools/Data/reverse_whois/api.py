import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_reverse_whois_with_owner_email(email: str, whois: str, page: str='1', mode: str='default', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get complete Whois Detail with Owner Email"
    email: search with owner email
        page: pass page value if whois records found more then one page
        mode: mini(upto 100 whois records per page) default(upto 50 whois records per page).
        format: available formats xml,json
        
    """
    url = f"https://reverse-whois7.p.rapidapi.com/whois"
    querystring = {'email': email, 'whois': whois, }
    if page:
        querystring['page'] = page
    if mode:
        querystring['mode'] = mode
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reverse-whois7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reverse_whois_with_owner_name(whois: str, owner: str, mode: str='default', page: str='1', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get complete Whois Detail with Owner Name"
    owner: search with owner name
        mode: mini(upto 100 whois records per page) default(upto 50 whois records per page).
        page: pass page value if whois records found more then one page
        format: available formats xml,json
        
    """
    url = f"https://reverse-whois7.p.rapidapi.com/whois"
    querystring = {'whois': whois, 'owner': owner, }
    if mode:
        querystring['mode'] = mode
    if page:
        querystring['page'] = page
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reverse-whois7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reverse_whois_with_owner_company(whois: str, company: str, format: str='json', mode: str='default', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get complete Whois Detail with Owner Company"
    company: search with owner company name
        format: available formats xml,json
        mode: mini(upto 100 whois records per page) default(upto 50 whois records per page).
        page: pass page value if whois records found more then one page
        
    """
    url = f"https://reverse-whois7.p.rapidapi.com/whois"
    querystring = {'whois': whois, 'company': company, }
    if format:
        querystring['format'] = format
    if mode:
        querystring['mode'] = mode
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reverse-whois7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reverse_whois_with_domain_keyword(keyword: str, whois: str, format: str='json', page: str='1', mode: str='default', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get complete Whois Detail with domain keyword"
    keyword: domain keyword
        format: available formats xml,json
        page: pass page value if whois records found more then one page
        mode: mini(upto 100 whois records per page) default(upto 50 whois records per page).
        
    """
    url = f"https://reverse-whois7.p.rapidapi.com/whois"
    querystring = {'keyword': keyword, 'whois': whois, }
    if format:
        querystring['format'] = format
    if page:
        querystring['page'] = page
    if mode:
        querystring['mode'] = mode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reverse-whois7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

