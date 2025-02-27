import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_arsenal(lang: str, category: str=None, name: str='ghost', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method can get you all Valorant Arsenal with different language also that allow to search by name and weapon category."
    
    """
    url = f"https://valorant-agents-maps-arsenal.p.rapidapi.com/arsenal/{lang}"
    querystring = {}
    if category:
        querystring['category'] = category
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "valorant-agents-maps-arsenal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_agents(lang: str, name: str='jett', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method can get all agents information with different languaage. You can also search by agent name."
    
    """
    url = f"https://valorant-agents-maps-arsenal.p.rapidapi.com/agents/{lang}"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "valorant-agents-maps-arsenal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_maps(lang: str, name: str='fracture', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method can get you all valorants maps information with different language also thats allows the search by name"
    
    """
    url = f"https://valorant-agents-maps-arsenal.p.rapidapi.com/maps/{lang}"
    querystring = {}
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "valorant-agents-maps-arsenal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

