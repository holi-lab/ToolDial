import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fixtures(compids: str='1,3', pagesize: str='20', clubids: str='1,2', page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When making a request to this API, it will return a JSON containing a list of upcoming matches in Premier League"
    
    """
    url = f"https://premier-league-upcoming-matches1.p.rapidapi.com/fixtures"
    querystring = {}
    if compids:
        querystring['compIds'] = compids
    if pagesize:
        querystring['pageSize'] = pagesize
    if clubids:
        querystring['clubIds'] = clubids
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-upcoming-matches1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions(abbr: str='EN_PR', name: str='Premier League', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will return an array of **5 biggest competitions** in which all clubs in* Premier League* are able to participate"
    
    """
    url = f"https://premier-league-upcoming-matches1.p.rapidapi.com/competitions"
    querystring = {}
    if abbr:
        querystring['abbr'] = abbr
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-upcoming-matches1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def clubs(abbr: str='ARS', name: str='Arsenal', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will supply client info about all clubs that are playing the First Premier League Competition"
    
    """
    url = f"https://premier-league-upcoming-matches1.p.rapidapi.com/clubs"
    querystring = {}
    if abbr:
        querystring['abbr'] = abbr
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premier-league-upcoming-matches1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

