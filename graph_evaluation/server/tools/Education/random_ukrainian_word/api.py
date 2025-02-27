import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getrandomwords(length: int=None, maxlength: int=6, excludes: str='кли', minlength: int=2, startswith: str='во', endswith: str='я', amount: int=1, includes: str='ол', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random Ukrainian words."
    length: Setting word exact length
        maxlength: Setting word maximum length
        excludes: Setting the letters which word should not include
        minlength: Setting word minimum length
        startswith: Setting the letters with which the word should begin
        endswith: Setting the letters with which the word should end
        amount: Setting amount of requested words
        includes: Setting the letters which word should include
        
    """
    url = f"https://random-ukrainian-word.p.rapidapi.com/words"
    querystring = {}
    if length:
        querystring['length'] = length
    if maxlength:
        querystring['maxLength'] = maxlength
    if excludes:
        querystring['excludes'] = excludes
    if minlength:
        querystring['minLength'] = minlength
    if startswith:
        querystring['startsWith'] = startswith
    if endswith:
        querystring['endsWith'] = endswith
    if amount:
        querystring['amount'] = amount
    if includes:
        querystring['includes'] = includes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "random-ukrainian-word.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

