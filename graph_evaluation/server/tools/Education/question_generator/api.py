import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def blank_fields(text: str, nbr: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this API to generate blank field questions from text. It uses AI to detect words to be replaced. Currently supporting only English."
    nbr: This is the number of questions to be generated. It's 5 by default
        
    """
    url = f"https://question-generator.p.rapidapi.com/"
    querystring = {'text': text, }
    if nbr:
        querystring['nbr'] = nbr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "question-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

