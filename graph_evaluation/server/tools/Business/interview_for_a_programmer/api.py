import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_right_answers(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get right answers by question id"
    
    """
    url = f"https://interview-for-a-programmer.p.rapidapi.com/api/right-answers/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interview-for-a-programmer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_question(levelname: str, languagename: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "type : THEORY,PRACTICE;
		languageName: JAVA,PHP,C,PYTHON;
		levelName: JUNIOR,MIDDLE,SENIOR;"
    
    """
    url = f"https://interview-for-a-programmer.p.rapidapi.com/api/questions"
    querystring = {'levelName': levelname, 'languageName': languagename, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "interview-for-a-programmer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

