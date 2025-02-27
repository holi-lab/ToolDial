import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_license_number(licensenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search US realtors database by License Number"
    
    """
    url = f"https://us-realtors-real-estate-agents.p.rapidapi.com/LicenseNumber"
    querystring = {'LicenseNumber': licensenumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-realtors-real-estate-agents.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_name(lastname: str, firstname: str, page: int, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search US realtors database by name
		State is optional"
    
    """
    url = f"https://us-realtors-real-estate-agents.p.rapidapi.com/SearchRealtor"
    querystring = {'LastName': lastname, 'FirstName': firstname, 'Page': page, }
    if state:
        querystring['State'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-realtors-real-estate-agents.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

