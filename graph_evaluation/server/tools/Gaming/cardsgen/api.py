import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_cards_randomly_generated(n: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cards randomly generated without replacement.
		A card is made of a `suite` among "hearts", "club", "diamonds", "spades" and a `value` among "2" ... "10" , "jack", "queen", "king", "ace""
    n: `n`  must be between 1 and 52 include
        
    """
    url = f"https://cardsgen.p.rapidapi.com/cards"
    querystring = {}
    if n:
        querystring['n'] = n
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cardsgen.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

