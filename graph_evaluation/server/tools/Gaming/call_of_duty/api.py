import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_modern_warfare_player_matches(gamer_tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets recent Modern Warfare multiplayer matches (up to last 20) of requested player."
    gamer_tag: The # is replaced by %23
        
    """
    url = f"https://call-of-duty8.p.rapidapi.com/modern-warfare/matches"
    querystring = {'gamer_tag': gamer_tag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "call-of-duty8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_modern_warfare_player_stats(gamer_tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats for Modern Warfare career and multiplayer modes of requested player."
    gamer_tag: Ensure you replace any '#' with '%23
        
    """
    url = f"https://call-of-duty8.p.rapidapi.com/modern-warfare/stats"
    querystring = {'gamer_tag': gamer_tag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "call-of-duty8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_warzone_player_matches(gamer_tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets recent Warzone matches (up to last 20) of requested player."
    gamer_tag: Ensure you replace any '#' with '%23
        
    """
    url = f"https://call-of-duty8.p.rapidapi.com/warzone/matches"
    querystring = {'gamer_tag': gamer_tag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "call-of-duty8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_warzone_player_stats(gamer_tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats for Warzone modes of requested player."
    gamer_tag: Ensure you replace  any '#' with '%23'
        
    """
    url = f"https://call-of-duty8.p.rapidapi.com/warzone/stats"
    querystring = {'gamer_tag': gamer_tag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "call-of-duty8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

