import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def highlights(username: str='kimkardashian', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Highlights"
    
    """
    url = f"https://ig-private-api.p.rapidapi.com/api/profile/highlights"
    querystring = {}
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ig-private-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def info(username: str='kimkardashian', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Info"
    
    """
    url = f"https://ig-private-api.p.rapidapi.com/api/profile/info"
    querystring = {}
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ig-private-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(username: str='kim', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    
    """
    url = f"https://ig-private-api.p.rapidapi.com/api/profile/search"
    querystring = {}
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ig-private-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publications(nextmaxid: str=None, username: str='kimkardashian', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Publications"
    
    """
    url = f"https://ig-private-api.p.rapidapi.com/api/profile/publications"
    querystring = {}
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ig-private-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stories(username: str='kimkardashian', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stories"
    
    """
    url = f"https://ig-private-api.p.rapidapi.com/api/profile/stories"
    querystring = {}
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ig-private-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def igtv(username: str='kimkardashian', nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IGTV"
    
    """
    url = f"https://ig-private-api.p.rapidapi.com/api/profile/igtv_channel"
    querystring = {}
    if username:
        querystring['username'] = username
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ig-private-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def highlight_items(highlightid: str='highlight:18064021144190483', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Highlight Items"
    
    """
    url = f"https://ig-private-api.p.rapidapi.com/api/profile/highlight_items"
    querystring = {}
    if highlightid:
        querystring['highlightId'] = highlightid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ig-private-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

