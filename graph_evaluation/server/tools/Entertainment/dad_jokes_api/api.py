import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dad_jokes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random dad joke to brighten up your day! This API returns a random dad joke from a list of pre-selected jokes. Jokes are updated regularly to keep the content fresh and funny. Whether you’re looking for a laugh or just need to break the ice, this API is the perfect source of cheesy dad humor."
    
    """
    url = f"https://dad-jokes-api1.p.rapidapi.com/dad_jokes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dad-jokes-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

