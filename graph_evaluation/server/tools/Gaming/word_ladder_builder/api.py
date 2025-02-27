import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def build_word_ladders(endword: str, startword: str, dictionary: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create Word Ladders between two words. By default it uses 2019 edition of Collins Dictionary."
    
    """
    url = f"https://word-ladder-builder.p.rapidapi.com/wordladder.php"
    querystring = {'EndWord': endword, 'StartWord': startword, }
    if dictionary:
        querystring['Dictionary'] = dictionary
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-ladder-builder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

