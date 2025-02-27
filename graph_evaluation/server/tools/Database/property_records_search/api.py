import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_name(lastname: str, firstname: str, state: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search property records by name"
    
    """
    url = f"https://property-records-search.p.rapidapi.com/SearchPeople"
    querystring = {'LastName': lastname, 'FirstName': firstname, }
    if state:
        querystring['State'] = state
    if page:
        querystring['Page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "property-records-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

