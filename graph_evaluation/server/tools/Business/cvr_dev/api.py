import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def r_cvr_virksomhed(navn: str=None, cvr_nummer: int=10103940, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dette endpoint returnerer alt data om en virksomhed, som den er i CVR. Dette dataformat er bestemt af Virk, ikke af cvr.dev."
    
    """
    url = f"https://cvr-dev.p.rapidapi.com/api/cvr/virksomhed"
    querystring = {}
    if navn:
        querystring['navn'] = navn
    if cvr_nummer:
        querystring['cvr_nummer'] = cvr_nummer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cvr-dev.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

