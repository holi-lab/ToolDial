import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def kred_activity(source: str, username: str, count: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the influence/outreach of recent activity for a user.  A 204 response means the user was not found."
    source: Source of the activity, e.g. twitter
        username: The username related to the source, e.g. bbcnews
        count: The number of items to return
        page: Which page to return
        
    """
    url = f"https://kred-exp-v2.p.rapidapi.com/kred/activity/{source}/{username}"
    querystring = {}
    if count:
        querystring['count'] = count
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kred-exp-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kred_score(source: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Kredentials presents the Kred score of a user in a comprehensive format that makes it simple to understand the source of their Kred score. Internally, we refer to this as an identity archive.  A 204 response means the user was not found."
    source: Source of the score, e.g. twitter
        username: The username related to the source, e.g. neilhimself
        
    """
    url = f"https://kred-exp-v2.p.rapidapi.com/kred/score/{source}/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kred-exp-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

