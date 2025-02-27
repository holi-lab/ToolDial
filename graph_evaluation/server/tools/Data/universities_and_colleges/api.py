import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def universities(search: str=None, includeuniversitydetails: bool=None, limit: int=10, countrycode: str='US', page: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of universities"
    search: Filter results by searching / autocompleting the name value.
        includeuniversitydetails: Include additional university details such as school colors, mascot, website, address, and more. Including details costs $0.01. Default value is false.
        limit: Default is 10. Max page size 50.
        countrycode: Ex/ US
        page: Default is 0.
        
    """
    url = f"https://universities-and-colleges.p.rapidapi.com/universities"
    querystring = {}
    if search:
        querystring['search'] = search
    if includeuniversitydetails:
        querystring['includeUniversityDetails'] = includeuniversitydetails
    if limit:
        querystring['limit'] = limit
    if countrycode:
        querystring['countryCode'] = countrycode
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "universities-and-colleges.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def university_details_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a single university by ID. Automatically includes any available University details (University details cost $0.01 if additional details like school colors, mascot, website, address, and more are present in response)."
    
    """
    url = f"https://universities-and-colleges.p.rapidapi.com/universities-by-id"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "universities-and-colleges.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

