import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_mailbox(address: str, public: bool=None, ttl: str='3600', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a new mailbox using specified email address"
    address: New mailbox address
        public: This parameter is processed in PRO, ULTRA and MEGA plans only. 
By default, it is set to false and a private mailbox is created. To create a public mailbox, set it to true.
All created mailboxes in FREE plan are public.








 
        ttl: Define mailbox lifetime in seconds (600-86400)
        
    """
    url = f"https://easy-fast-temp-mail.p.rapidapi.com/create"
    querystring = {'address': address, }
    if public:
        querystring['public'] = public
    if ttl:
        querystring['ttl'] = ttl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-fast-temp-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_email_content(address: str, message_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get content of specified message-id"
    
    """
    url = f"https://easy-fast-temp-mail.p.rapidapi.com/{address}/{message_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-fast-temp-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_emails_list(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of received email in mailbox"
    
    """
    url = f"https://easy-fast-temp-mail.p.rapidapi.com/{address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-fast-temp-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot(address: str, message_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Render email contents and return a PNG image"
    
    """
    url = f"https://easy-fast-temp-mail.p.rapidapi.com/{address}/{message_id}/screenshot"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-fast-temp-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def attachments(message_id: str, address: str, idx: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve email attachment list or a single attachment file.
		If query parameter **idx** is missing, a JSON payload with email attachments details is returned.
		If query parameter **idx**  is present, content of attachment  file is returned."
    idx: Use this parameter to retrieve attachment contents. IDX = 0 is the first attachment file.
        
    """
    url = f"https://easy-fast-temp-mail.p.rapidapi.com/{address}/{message_id}/attachments"
    querystring = {}
    if idx:
        querystring['idx'] = idx
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-fast-temp-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domain_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return a list of your custom domains. For each domain, verification status and token is reported."
    
    """
    url = f"https://easy-fast-temp-mail.p.rapidapi.com/domains"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-fast-temp-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def click_on_links(message_id: str, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Click on email html body links  and return results."
    
    """
    url = f"https://easy-fast-temp-mail.p.rapidapi.com/{address}/{message_id}/click"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easy-fast-temp-mail.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

