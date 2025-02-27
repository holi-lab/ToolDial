import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_answer_from_magic_8_ball(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tired of making decisions on your own? Bored of plain old magic 8-balls? Introducing the all-new hilarious Magic 8-Ball API! Get ready to laugh out loud as you receive wacky responses to your burning questions. Will your crush finally notice you? Should you eat that second slice of cake? The Magic 8-Ball API has the answers you never knew you needed."
    
    """
    url = f"https://magic-8-ball4.p.rapidapi.com/magic_8_ball"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magic-8-ball4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

