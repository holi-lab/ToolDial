import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def name_search(lastname: str, state: str, firstname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search business licenses by name"
    
    """
    url = f"https://business-license-search-us.p.rapidapi.com/search_Person"
    querystring = {'LastName': lastname, 'state': state, 'FirstName': firstname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-license-search-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def license_number_search(state: str, licensenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search business licenses by license number and state"
    
    """
    url = f"https://business-license-search-us.p.rapidapi.com/search_LicenseNumber"
    querystring = {'state': state, 'LicenseNumber': licensenumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-license-search-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_name_search(company: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search business licenses by company name and state"
    
    """
    url = f"https://business-license-search-us.p.rapidapi.com/search_Company"
    querystring = {'Company': company, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "business-license-search-us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

