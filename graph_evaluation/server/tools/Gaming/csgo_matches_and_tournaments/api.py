import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def played(content_type: str='application/json', limit: int=10, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Last of played cs:go matches"
    
    """
    url = f"https://csgo-matches-and-tournaments.p.rapidapi.com/matches"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "csgo-matches-and-tournaments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upcoming(content_type: str='application/json', page: int=1, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Last of upcoming cs:go matches"
    
    """
    url = f"https://csgo-matches-and-tournaments.p.rapidapi.com/upcoming_matches"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "csgo-matches-and-tournaments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

