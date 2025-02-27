import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def regular_die(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Roll one regular die."
    
    """
    url = f"https://dice-roll-simulator.p.rapidapi.com/regular-die"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dice-roll-simulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_die(sides: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Roll one die with any number of sides."
    
    """
    url = f"https://dice-roll-simulator.p.rapidapi.com/custom-die"
    querystring = {}
    if sides:
        querystring['sides'] = sides
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dice-roll-simulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regular_dice(dice: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Roll any number of regular dice."
    
    """
    url = f"https://dice-roll-simulator.p.rapidapi.com/regular-dice"
    querystring = {}
    if dice:
        querystring['dice'] = dice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dice-roll-simulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_dice(dice: int=3, sides: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Roll any number of dice with any number of sides."
    
    """
    url = f"https://dice-roll-simulator.p.rapidapi.com/custom-dice"
    querystring = {}
    if dice:
        querystring['dice'] = dice
    if sides:
        querystring['sides'] = sides
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dice-roll-simulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regular_dice_rolls(rolls: int=2, dice: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Roll any number of regular dice any number of times."
    
    """
    url = f"https://dice-roll-simulator.p.rapidapi.com/regular-dice-rolls"
    querystring = {}
    if rolls:
        querystring['rolls'] = rolls
    if dice:
        querystring['dice'] = dice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dice-roll-simulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_dice_rolls(dice: int=3, sides: int=8, rolls: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Roll any number of dice with any number of sides any number of times."
    
    """
    url = f"https://dice-roll-simulator.p.rapidapi.com/custom-dice-rolls"
    querystring = {}
    if dice:
        querystring['dice'] = dice
    if sides:
        querystring['sides'] = sides
    if rolls:
        querystring['rolls'] = rolls
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dice-roll-simulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

