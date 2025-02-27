import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def us(cards: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "US game spec.
		Returns numbers between 1 and 75, randomized in 5 groups.
		
		If you have a paid plan you can specify the number of cards to generate as a query parameter: `?cards=10`
		This will return an array of array (the cards).
		
		Please note that empty spaces are not returned as it is more portable for who don't need them. You are free to loop through the cards and add them at your wish."
    
    """
    url = f"https://bingoapi.p.rapidapi.com/us"
    querystring = {}
    if cards:
        querystring['cards'] = cards
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bingoapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eu(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "EU game spec.
		Returns numbers between 1 and 90.
		
		If you have a paid plan you can specify the number of cards to generate as a query parameter: `?cards=10`
		This will return an array of array (the cards).
		
		Please note that empty spaces are not returned as it is more portable for who don't need them. You are free to loop through the cards and add them at your wish."
    
    """
    url = f"https://bingoapi.p.rapidapi.com/eu"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bingoapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

