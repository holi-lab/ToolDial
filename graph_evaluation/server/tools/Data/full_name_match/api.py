import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_full_name_similarity_key(fullname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a similarity key for fuzzy matching with other full name records and data"
    fullname: full name input parameter
        
    """
    url = f"https://full-name-match.p.rapidapi.com/getfullnamematch"
    querystring = {'fullname': fullname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "full-name-match.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

