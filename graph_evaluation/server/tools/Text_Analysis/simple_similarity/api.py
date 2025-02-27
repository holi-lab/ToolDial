import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def entrypoint(txt2: str, txt1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the main and only endpoint. Here you make the full text call including both texts (text #1 and text #2)."
    
    """
    url = f"https://simple-similarity.p.rapidapi.com/"
    querystring = {'txt2': txt2, 'txt1': txt1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-similarity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

