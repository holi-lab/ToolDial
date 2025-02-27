import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_full_name_parsed_match(firstname: str, lastname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a similarity key for fuzzy matching with other full name records and data"
    firstname: First name only of full name
        lastname: Last name only of full name
        
    """
    url = f"https://parsed-name-match.p.rapidapi.com/getfullnameparsedmatch"
    querystring = {'firstname': firstname, 'lastname': lastname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "parsed-name-match.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

