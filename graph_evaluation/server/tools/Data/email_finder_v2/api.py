import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_by_domain(domain: str, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find multiple email addresses by searching by domain.
		
		The API returns 5 email addresses per request. You can off set ``offSet`` if you need more emails."
    
    """
    url = f"https://email-finder7.p.rapidapi.com/email-address/find-many-domain"
    querystring = {'domain': domain, }
    if offset:
        querystring['offSet'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-finder7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_email(domain: str, personlastname: str, personfirstname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find any email address anywhere in the world"
    
    """
    url = f"https://email-finder7.p.rapidapi.com/email-address/find-one/"
    querystring = {'domain': domain, 'personLastName': personlastname, 'personFirstName': personfirstname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-finder7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

