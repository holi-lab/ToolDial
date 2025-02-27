import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def greet_me_cuz_it_s_my_birthday_and_i_m_lonely(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Greets you a happy birthday with a fun lie, not fun fact cuz it is overused."
    
    """
    url = f"https://happy-birthday-api.p.rapidapi.com/prod/programmers-birthday"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "happy-birthday-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

