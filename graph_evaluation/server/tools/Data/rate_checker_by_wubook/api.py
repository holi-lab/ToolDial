import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def monitored_competitors_from_a_given_user(user_id: str='auto', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-rate-checker-by-wubook.p.rapidapi.com/user"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-rate-checker-by-wubook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def snapshots_available_from_a_given_stay(stay_id: str='539dba784e56e1686bbb0f41', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-rate-checker-by-wubook.p.rapidapi.com/stay"
    querystring = {}
    if stay_id:
        querystring['stay_id'] = stay_id
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-rate-checker-by-wubook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stays_from_a_given_competitor(competitor_id: str='539dba784e56e1686bbb0f40', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-rate-checker-by-wubook.p.rapidapi.com/competitor"
    querystring = {}
    if competitor_id:
        querystring['competitor_id'] = competitor_id
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-rate-checker-by-wubook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

