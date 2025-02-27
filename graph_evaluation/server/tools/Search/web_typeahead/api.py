import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, cc: str='us', setlang: str='en', count: str='6', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Typeahead. Autocomplete."
    cc: country_code
        setlang: language code
        
    """
    url = f"https://web-typeahead.p.rapidapi.com/search"
    querystring = {'q': q, }
    if cc:
        querystring['cc'] = cc
    if setlang:
        querystring['setLang'] = setlang
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-typeahead.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

