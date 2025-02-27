import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_essays(source_type: str, mentor: str='writing', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "source_type : mentor or writing"
    source_type: source_type : writing or mentor
        
    """
    url = f"https://ielts-writing-essays.p.rapidapi.com/ielts-writing/{source_type}/task-2"
    querystring = {}
    if mentor:
        querystring['mentor'] = mentor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ielts-writing-essays.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

