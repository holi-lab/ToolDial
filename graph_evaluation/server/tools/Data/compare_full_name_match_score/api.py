import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_full_name_match_score(fullname2: str, fullname1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes two person names and returns a score from 0-100 on the likelihood that they are a match. Can be used with other criteria for identifying duplicate data."
    
    """
    url = f"https://compare-full-name-match-score.p.rapidapi.com/getfullnamematchscore"
    querystring = {'fullname2': fullname2, 'fullname1': fullname1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "compare-full-name-match-score.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

