import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def words2number(query: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the corresponding number for your input sentence. Specify the language with the associated request parameter (currently available for italian ("it") and english ("en"))"
    
    """
    url = f"https://words-to-numbers.p.rapidapi.com/wordsToNumber"
    querystring = {'query': query, 'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "words-to-numbers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

