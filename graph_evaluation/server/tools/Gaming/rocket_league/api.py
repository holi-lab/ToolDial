import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tournaments(region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the scheduled tournaments:
		`rocket-league1.p.rapidapi.com/tournaments/:region`"
    region: The region to search. Valid options include:
- `asia-east`
- `asia-se-mainland`
- `asia-se-maritime`
- `europe`
- `india`
- `asia-east`
- `middle-east`
- `oceania`
- `south-africa`
- `south-america`
- `us-east`
- `us-west`
        
    """
    url = f"https://rocket-league1.p.rapidapi.com/tournaments/{region}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenges(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the seasonal and weekly challenges."
    
    """
    url = f"https://rocket-league1.p.rapidapi.com/challenges/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the Esports events:
		`rocket-league1.p.rapidapi.com/esports`"
    
    """
    url = f"https://rocket-league1.p.rapidapi.com/esports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shops(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the current shops:
		`rocket-league1.p.rapidapi.com/shops/:type`"
    
    """
    url = f"https://rocket-league1.p.rapidapi.com/shops/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the news:
		`rocket-league1.p.rapidapi.com/news`"
    
    """
    url = f"https://rocket-league1.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def population(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the playlist population:
		`rocket-league1.p.rapidapi.com/population`"
    
    """
    url = f"https://rocket-league1.p.rapidapi.com/population"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blog(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the blog:
		`rocket-league1.p.rapidapi.com/blog`"
    
    """
    url = f"https://rocket-league1.p.rapidapi.com/blog"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def titles(player: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check a player's titles:
		`rocket-league1.p.rapidapi.com/titles/:player`"
    player: The Epic Games account ID or display name to search.
        
    """
    url = f"https://rocket-league1.p.rapidapi.com/titles/{player}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile(player: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check a player's profile:
		`rocket-league1.p.rapidapi.com/profile/:player`"
    player: The Epic Games account ID or display name to search.
        
    """
    url = f"https://rocket-league1.p.rapidapi.com/profile/{player}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stat(player: str, stat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check a player's stat:
		`rocket-league1.p.rapidapi.com/stat/:player/:stat`"
    player: The Epic Games account ID or display name to search.
        stat: The stat to search. Valid options include:
- `assists`
- `goals`
- `mvps`
- `saves`
- `shots`
- `wins`
        
    """
    url = f"https://rocket-league1.p.rapidapi.com/stat/{player}/{stat}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ranks(player: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check a player's ranks:
		`rocket-league1.p.rapidapi.com/ranks/:player`"
    player: The Epic Games account ID or display name to search.
        
    """
    url = f"https://rocket-league1.p.rapidapi.com/ranks/{player}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club(player: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check a player's club:
		`rocket-league1.p.rapidapi.com/club/:player`"
    player: The Epic Games account ID or display name to search.
        
    """
    url = f"https://rocket-league1.p.rapidapi.com/club/{player}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rocket-league1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

