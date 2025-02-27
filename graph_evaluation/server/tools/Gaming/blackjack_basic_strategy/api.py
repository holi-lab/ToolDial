import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def true_count(decksremaining: int, runningcount: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to determine the True Count of the current card shoe. It takes in the running count of the game and the number of decks remaining in play and outputs the true count as a number."
    
    """
    url = f"https://blackjack-basic-strategy.p.rapidapi.com/trueCount/{runningcount}/{decksremaining}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blackjack-basic-strategy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pairs(dealerupcard: int, pairof: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is to be used when the player has a pair. With this endpoint, the total value of the player's cards are not to be input, but the value of the card which there are two of. For example, when the player has a Queen and a Jack, the pairOf parameter should be 10."
    
    """
    url = f"https://blackjack-basic-strategy.p.rapidapi.com/pair/{pairof}/{dealerupcard}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blackjack-basic-strategy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soft_totals(dealerupcard: int, playertotal: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is to be used when one has an Ace which represents an 11 EXCEPT for when one has a pair of Aces (See 'Pairs' endpoint). If one does not have an Ace which represents an 11, the 'Hard Totals' endpoint should be used."
    
    """
    url = f"https://blackjack-basic-strategy.p.rapidapi.com/soft/{playertotal}/{dealerupcard}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blackjack-basic-strategy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hard_totals(playertotal: int, dealerupcard: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the endpoint to use when one does not have a pair (See 'Pairs' endpoint) or an Ace that represents as 11 (See 'Soft Totals' endpoint). For example, if one has a 5 and a 7 and the dealer has a 3."
    
    """
    url = f"https://blackjack-basic-strategy.p.rapidapi.com/hard/{playertotal}/{dealerupcard}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blackjack-basic-strategy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

