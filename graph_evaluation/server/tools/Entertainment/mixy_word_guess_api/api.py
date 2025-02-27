import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def produce_mixy_words_list(amount: int=10, difficulty: str='easy', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It produces random words in a requested amount and varios difficulty levels such as; "easy", "medium", "hard". In the query, "amount" and "difficulty" are optional parameters."
    amount: Valid values: integer between 1 and 100
        difficulty: valid values: "easy", "medium", "hard"
        
    """
    url = f"https://mixy-word-guess-api.p.rapidapi.com/api/v1/word"
    querystring = {}
    if amount:
        querystring['amount'] = amount
    if difficulty:
        querystring['difficulty'] = difficulty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mixy-word-guess-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

