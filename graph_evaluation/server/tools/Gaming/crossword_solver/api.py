import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cross(word: str, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Using this API is easy! An example?**
		
		Search string:
		- must contain letters (a-z) AND at least one _ (underscore) for unknown letters.
		- must have a minimum length of 3 and a maximum lenght of 65 characters.
		- is a required parameter.
		
		Example request:   `word=u_b_l_e_a_l_`
		
		Language:
		- en, es, de (English, Spanish, German)
		- is a optional parameter. If not set /en/ is used. 
		
		Example request:   `lang=en`
		
		**This is fun!**"
    
    """
    url = f"https://crossword-solver.p.rapidapi.com/cross"
    querystring = {'word': word, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crossword-solver.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

