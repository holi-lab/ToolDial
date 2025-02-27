import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cryptogram(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a cryptogram of the string passed via the "s" parameter.
		
		Any alphabetical characters are modified using a simple substitution cipher. No alphabetical character will ever map to itself.
		
		Any non-alphabetical characters (punctuation, numbers, etc.) will remain unchanged in the returned cryptogram string."
    
    """
    url = f"https://cryptogram-generator.p.rapidapi.com/"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptogram-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

