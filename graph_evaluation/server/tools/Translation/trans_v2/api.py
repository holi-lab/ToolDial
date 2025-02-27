import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translation_suppliers(is_id: str, yourquestions: str='up to 500 characters', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We Welcome All Your Needs for Translation in any field"
    id: volume starting from 1000 words
        yourquestions: Freely Ask Your Questions for your projects
        
    """
    url = f"https://trans2.p.rapidapi.com/suppliers/{is_id}"
    querystring = {}
    if yourquestions:
        querystring['YourQuestions'] = yourquestions
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trans2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

