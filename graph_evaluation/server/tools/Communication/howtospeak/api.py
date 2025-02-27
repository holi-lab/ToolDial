import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def english2chinese(text: str, notrans: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translate your English into Chinese character phonetically."
    text: The text you want to translate. The value should be Chinese (when notrans=0) or English (when notrans=1)
        notrans: Whether translate input text into Chinese first: Translate it (notrans = 0) or Don't translate it (notrans = 1)
        
    """
    url = f"https://howtospeak.p.rapidapi.com/api/e2c"
    querystring = {'text': text, }
    if notrans:
        querystring['notrans'] = notrans
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "howtospeak.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def japanese2chinese(text: str, notrans: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Translate Japanese into Chinese character phonetically."
    text: The text you want to translate. The value should be Chinese (when notrans=0) or Japanese (when notrans=1)
        notrans: Whether translate input text into Chinese first: Translate it (notrans = 0) or Don't translate it (notrans = 1)
        
    """
    url = f"https://howtospeak.p.rapidapi.com/api/j2c"
    querystring = {'text': text, }
    if notrans:
        querystring['notrans'] = notrans
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "howtospeak.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

