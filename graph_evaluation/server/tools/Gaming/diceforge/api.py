import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def roll(sides: int, count: int=1, advantage: bool=None, modifier: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rolls a set of dice and returns the results.
		
		Methods: GET, POST
		
		Parameters:
		
		sides (integer): A list of integers representing the number of sides for each die.
		count (integer): A list of integers representing the number of times to roll each die. Defaults to 1 if not provided.
		modifier (integer): A list of integers representing the modifier to apply to the total of each die roll.
		advantage (boolean): If set to true, rolls each die twice and takes the higher result.
		disadvantage (boolean): If set to true, rolls each die twice and takes the lower result."
    
    """
    url = f"https://diceforge.p.rapidapi.com/roll"
    querystring = {'sides': sides, }
    if count:
        querystring['count'] = count
    if advantage:
        querystring['advantage'] = advantage
    if modifier:
        querystring['modifier'] = modifier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diceforge.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statroll(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rolls 4d6 for generating character stats, dropping the lowest die result from the total, and returns the results.
		
		Methods: GET, POST
		
		Parameters:
		
		modifier (integer): An integer representing the modifier to apply to the total of the stat roll."
    
    """
    url = f"https://diceforge.p.rapidapi.com/statroll"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diceforge.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

