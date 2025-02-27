import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def motivational_quotes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The motivational quotes endpoint provides users with inspiring quotes from notable figures that can help motivate and encourage them to achieve their goals. Each quote is accompanied by the name of the author or speaker, giving users the opportunity to learn from and be inspired by some of the greatest minds in history. These quotes are designed to uplift and empower users, reminding them that they have the strength and ability to overcome any obstacle and accomplish their dreams."
    
    """
    url = f"https://motivational-quotes-quotable-api.p.rapidapi.com/motivational_quotes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motivational-quotes-quotable-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

