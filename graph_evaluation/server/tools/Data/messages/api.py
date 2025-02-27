import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmsgs(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetMsg"
    category: Select a category to get unlimited messages.Eg: Love,quotes,friendship,Good night,Good morning,funny,Birthday,Sad,Sweet,Random
        
    """
    url = f"https://ajith-messages.p.rapidapi.com/getMsgs"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ajith-messages.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

