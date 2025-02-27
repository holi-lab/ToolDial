import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def email_search(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to search by email address"
    
    """
    url = f"https://usa-people-search-public-records.p.rapidapi.com/SearchPeopleEmail"
    querystring = {'Email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "usa-people-search-public-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_search(phone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to search by a phone number"
    
    """
    url = f"https://usa-people-search-public-records.p.rapidapi.com/SearchPeoplePhone"
    querystring = {'Phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "usa-people-search-public-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def name_search(firstname: str, page: int, lastname: str, state: str='CA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to search by name. State is opotional"
    
    """
    url = f"https://usa-people-search-public-records.p.rapidapi.com/SearchPeople"
    querystring = {'FirstName': firstname, 'Page': page, 'LastName': lastname, }
    if state:
        querystring['State'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "usa-people-search-public-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

