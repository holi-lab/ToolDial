import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ublacklist_txt(game: str='minecraft', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ublacklist.txt endpoint"
    
    """
    url = f"https://stopmodreposts-blocklist.p.rapidapi.com/{game}/ublacklist.txt"
    querystring = {}
    if game:
        querystring['game'] = game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stopmodreposts-blocklist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hosts_txt(game: str='minecraft', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hosts file endpoint"
    
    """
    url = f"https://stopmodreposts-blocklist.p.rapidapi.com/{game}/hosts.txt"
    querystring = {}
    if game:
        querystring['game'] = game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stopmodreposts-blocklist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def txt(game: str='minecraft', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Plain text endpoint"
    
    """
    url = f"https://stopmodreposts-blocklist.p.rapidapi.com/{game}/sites.txt"
    querystring = {}
    if game:
        querystring['game'] = game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stopmodreposts-blocklist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yaml(game: str='minecraft', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "YAML Ain't Markup Language endpoint"
    
    """
    url = f"https://stopmodreposts-blocklist.p.rapidapi.com/{game}/sites.yaml"
    querystring = {}
    if game:
        querystring['game'] = game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stopmodreposts-blocklist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xml(game: str='minecraft', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extensible Markup Language endpoint"
    
    """
    url = f"https://stopmodreposts-blocklist.p.rapidapi.com/{game}/sites.xml"
    querystring = {}
    if game:
        querystring['game'] = game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stopmodreposts-blocklist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def json(game: str='minecraft', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "JavaScript Object Notation endpoint"
    
    """
    url = f"https://stopmodreposts-blocklist.p.rapidapi.com/{game}/sites.json"
    querystring = {}
    if game:
        querystring['game'] = game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stopmodreposts-blocklist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

