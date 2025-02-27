import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def code_for_you(codeisfun: str, codeforyou: str, codewithme: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Code For You"
    
    """
    url = f"https://code-for-you.p.rapidapi.com/"
    querystring = {'CodeIsFun': codeisfun, 'CodeForYou': codeforyou, 'CodeWithMe': codewithme, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "code-for-you.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

