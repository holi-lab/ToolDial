import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_list_of_games(console_name: str='NES', full_name: str='Super Mario Bros', region: str='NTSC', name_contains: str='Mario', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one or several games, with their name, console, region, and prices (loose, complete in box, and new in box)."
    
    """
    url = f"https://video-game-price.p.rapidapi.com/game"
    querystring = {}
    if console_name:
        querystring['console_name'] = console_name
    if full_name:
        querystring['full_name'] = full_name
    if region:
        querystring['region'] = region
    if name_contains:
        querystring['name_contains'] = name_contains
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_consoles_by_region(region: str='NTSC', console_name: str='NES', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one or several combinations of console and region."
    
    """
    url = f"https://video-game-price.p.rapidapi.com/consolebyregion"
    querystring = {}
    if region:
        querystring['region'] = region
    if console_name:
        querystring['console_name'] = console_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_regions(region: str='NTSC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one or several regions."
    
    """
    url = f"https://video-game-price.p.rapidapi.com/region"
    querystring = {}
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_consoles(console_name: str='NES', brand: str='Nintendo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one or several consoles, including their name and brand."
    
    """
    url = f"https://video-game-price.p.rapidapi.com/console"
    querystring = {}
    if console_name:
        querystring['console_name'] = console_name
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "video-game-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

