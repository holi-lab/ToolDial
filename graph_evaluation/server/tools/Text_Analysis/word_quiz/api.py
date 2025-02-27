import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def type_1(level: int, area: str='sat', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Customized word association quiz for game and e-learning software."
    level: Select the difficulty level of the test (1 - 10)
        area: Select a test to generate quiz questions and answers (es, ms, hs, ksat, toeic, toefl, teps, sat, ielts, gre, gmat, overall)
        
    """
    url = f"https://twinword-word-association-quiz.p.rapidapi.com/type1/"
    querystring = {'level': level, }
    if area:
        querystring['area'] = area
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-word-association-quiz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

