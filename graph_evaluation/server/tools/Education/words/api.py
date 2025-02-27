import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def unscramble(l: str, l2: str='spa', letters: str='populars', length: str='7', all: str='po', starting: str='p', ending: str='r', containing: str='pop', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Various word games"
    
    """
    url = f"https://words36.p.rapidapi.com/unscrambler-api"
    querystring = {'l': l, }
    if l2:
        querystring['l2'] = l2
    if letters:
        querystring['letters'] = letters
    if length:
        querystring['length'] = length
    if all:
        querystring['all'] = all
    if starting:
        querystring['starting'] = starting
    if ending:
        querystring['ending'] = ending
    if containing:
        querystring['containing'] = containing
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "words36.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

