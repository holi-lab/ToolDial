import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def entry(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve data related to the specified entry, such as word or phrasal verb. The data includes meanings, pronunciations, lemmas, normalized lemmas etc."
    entry: Entry (word, phrasal verb etc., case-sensitive)
        
    """
    url = f"https://lingua-robot.p.rapidapi.com/language/v1/entries/en/{entry}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lingua-robot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

