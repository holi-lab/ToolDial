import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    text: єηтєя уσυя тєχт αη∂ ¢нαηgє тσ ƒαη¢у тєχт υѕιηg тнιѕ ¢σσℓ αρι :
        
    """
    url = f"https://ajith-Fancy-text-v1.p.rapidapi.com/text"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ajith-Fancy-text-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

