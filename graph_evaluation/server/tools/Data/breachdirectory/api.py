import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def index(term: str, func: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Base URL for using each of BreachDirectory's API functions: auto, sources, and password."
    term: Enter the proper search term given the function you selected.
        func: Select the function you wish to use from the following:
- auto: Returns passwords, SHA-1 hashes, and sources given any username or email.
- sources: Returns sources given username or email.
- password: Returns how many times a given password has been leaked.
- domain: Returns passwords, SHA-1 hashes, and sources given any domain (Limited to 1000 results for security).
- dehash: Attempts to decrypt a given hash.
        
    """
    url = f"https://breachdirectory.p.rapidapi.com/"
    querystring = {'term': term, 'func': func, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "breachdirectory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

