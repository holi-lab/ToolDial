import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_questions(code: str='XaQbqQLneDPhf1q6xsCfEyfFCXa7QSLzNYvVhjjbFIETYQbhgAzqRw==', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all quiz questions and answer them one by one by POSTing them to the other endpoint"
    
    """
    url = f"https://easter-quick-quiz.p.rapidapi.com/EasterQuickQuiz"
    querystring = {}
    if code:
        querystring['code'] = code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "easter-quick-quiz.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

