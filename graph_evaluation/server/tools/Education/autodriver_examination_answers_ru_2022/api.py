import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getanswer(q: int, t: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get answers"
    q: Question number
        t: Ticket number
        
    """
    url = f"https://autodriver-examination-answers-ru-2022.p.rapidapi.com/getAnswer"
    querystring = {'q': q, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autodriver-examination-answers-ru-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

