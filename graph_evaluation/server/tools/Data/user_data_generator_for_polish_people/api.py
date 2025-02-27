import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_users_from_specific_year(count: int=5, year: int=1995, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return specific count of users from given year."
    
    """
    url = f"https://user-data-generator-for-polish-people.p.rapidapi.com/api/users/single-year"
    querystring = {}
    if count:
        querystring['count'] = count
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "user-data-generator-for-polish-people.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users_from_different_years(since: int=1970, until: int=2010, count: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return specific count of users from given range of years"
    
    """
    url = f"https://user-data-generator-for-polish-people.p.rapidapi.com/api/users"
    querystring = {}
    if since:
        querystring['since'] = since
    if until:
        querystring['until'] = until
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "user-data-generator-for-polish-people.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

