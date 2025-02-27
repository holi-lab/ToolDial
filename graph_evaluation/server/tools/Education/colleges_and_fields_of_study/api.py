import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(locality: str='east lansing', region: str='MI', institutionquery: str='state', page: int=1, fieldofstudyquery: str='computer', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for colleges by name, field of study, city and state."
    
    """
    url = f"https://colleges-and-fields-of-study.p.rapidapi.com/search"
    querystring = {}
    if locality:
        querystring['locality'] = locality
    if region:
        querystring['region'] = region
    if institutionquery:
        querystring['institutionQuery'] = institutionquery
    if page:
        querystring['page'] = page
    if fieldofstudyquery:
        querystring['fieldOfStudyQuery'] = fieldofstudyquery
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "colleges-and-fields-of-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

