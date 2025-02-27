import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def contacts_outreach(key: str='9gZJadaH4r', company: str=None, location: str=None, person: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API fetches the email and phone numbers of any given Person/Company/Market-Places' shops/ local vendors and much more."
    company: Any Organization/Company/Vendor/Marketplace/Support/Shop/Business/Brand
        location: Location of respective Person or Company
        person: Any Person that could be searched on Google.
        
    """
    url = f"https://email-finder9.p.rapidapi.com/get-contacts/"
    querystring = {}
    if key:
        querystring['key'] = key
    if company:
        querystring['company'] = company
    if location:
        querystring['location'] = location
    if person:
        querystring['person'] = person
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-finder9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_email(email: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify if the email does exist or not."
    
    """
    url = f"https://email-finder9.p.rapidapi.com/validate-email/"
    querystring = {'email': email, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "email-finder9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

