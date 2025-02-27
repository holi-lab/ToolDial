import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def word_of_the_day(x_agarathi_api_secret: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Tamil word of the day"
    x_agarathi_api_secret: Agrathi API key ( Get yours here http://agarathi.com/api/dictionary )
        
    """
    url = f"https://tamil.p.rapidapi.com/dictionary/wod"
    querystring = {'X-Agarathi-Api-Secret': x_agarathi_api_secret, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tamil.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

