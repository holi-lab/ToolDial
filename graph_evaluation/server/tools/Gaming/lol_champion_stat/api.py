import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def counter_stat(champ: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "display a list of champs that has the win rate lower than 49% for a particular champ.
		For character like K'Sante with quote inside name, use double quote instead of single quote"
    
    """
    url = f"https://lol-champion-stat.p.rapidapi.com/counter_stat"
    querystring = {'champ': champ, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lol-champion-stat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def champ_stat(champ: str, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "check champion basic stat
		each role(if available) will have its own data
		For character like K'Sante with quote inside name, use double quote instead of single quote"
    
    """
    url = f"https://lol-champion-stat.p.rapidapi.com/champ_stat/"
    querystring = {'champ': champ, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lol-champion-stat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def win_rate_ranking(role: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "check champion ranking
		'support', 'adc', 'jungle', 'mid', 'top'"
    
    """
    url = f"https://lol-champion-stat.p.rapidapi.com/champ_stat/ranking"
    querystring = {}
    if role:
        querystring['role'] = role
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lol-champion-stat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

