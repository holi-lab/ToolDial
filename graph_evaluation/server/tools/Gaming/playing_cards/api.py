import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random_cards(joker: str='enable', count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Random cards, defualt is all 52 cards.  Can include jokers (jokers=enable).  Can be selection of cards, ie count=10 or count=65"
    
    """
    url = f"https://playing-cards.p.rapidapi.com/api/randomcards"
    querystring = {}
    if joker:
        querystring['joker'] = joker
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "playing-cards.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def back_of_cards(colour: str='red', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Back of cards.  Colour = can be blue or red"
    
    """
    url = f"https://playing-cards.p.rapidapi.com/api/backcards"
    querystring = {}
    if colour:
        querystring['colour'] = colour
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "playing-cards.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deck_of_cards(decks: str='2', jokers: str='enable', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Deck of cards.  Jokers can be enabled (jokers=enable). You can have many decks ie decks=3"
    
    """
    url = f"https://playing-cards.p.rapidapi.com/api/deck"
    querystring = {}
    if decks:
        querystring['decks'] = decks
    if jokers:
        querystring['jokers'] = jokers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "playing-cards.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

