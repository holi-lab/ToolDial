import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_collection_of_emails(page: str='3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of emails from random domains on our list.
		
		**Parameters**
		->  page (For pagination purpose... i.e: page=4)"
    
    """
    url = f"https://email-finder3.p.rapidapi.com/getemails"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_email_by_domain(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of emails from any domain.
		
		**Parameters**
		-> url (Do not add any https, http or www. Must contain only the url without any https. i.e: google.com)"
    
    """
    url = f"https://email-finder3.p.rapidapi.com/searchemails"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

