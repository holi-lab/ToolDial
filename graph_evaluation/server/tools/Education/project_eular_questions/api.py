import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_50_questions(top_50_question: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will fetch questions top 50 questions from projecteuler.net"
    
    """
    url = f"https://project-eular-questions.p.rapidapi.com/top-50-question"
    querystring = {}
    if top_50_question:
        querystring['top-50-question'] = top_50_question
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "project-eular-questions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_question_from_project_eular(no: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will give detail of a question from project eular"
    
    """
    url = f"https://project-eular-questions.p.rapidapi.com/question"
    querystring = {'no': no, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "project-eular-questions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

