import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def card(suit: str='all', value: str='A', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Draw one card from a regular 52-card playing deck."
    
    """
    url = f"https://card-draw-simulator.p.rapidapi.com/card"
    querystring = {}
    if suit:
        querystring['suit'] = suit
    if value:
        querystring['value'] = value
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "card-draw-simulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cards(value: str='all', suit: str='♠', back: str='false', pulls: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Draw multiple cards from a regular 52-card playing deck."
    
    """
    url = f"https://card-draw-simulator.p.rapidapi.com/cards"
    querystring = {}
    if value:
        querystring['value'] = value
    if suit:
        querystring['suit'] = suit
    if back:
        querystring['back'] = back
    if pulls:
        querystring['pulls'] = pulls
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "card-draw-simulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

