import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns all the currently available countries.  Every item includes the following parameters: 
		- Short name
		- Short name lowcase
		- Full name
		- ISO alpha-2-code
		- ISO alpha-3-code
		- ISO numeric 3-digit code
		- Flag (from Twitter emojis)
		- Official flag —only for those countries which have an unnoficial flag"
    
    """
    url = f"https://rest-countries10.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rest-countries10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

