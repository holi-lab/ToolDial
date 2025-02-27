import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scuba_diving_emergency_questions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return 5 questions and answers from Scuba diving "emergency" category."
    
    """
    url = f"https://scuba-diving-quiz-version-101.p.rapidapi.com/trivia/emergency"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scuba-diving-quiz-version-101.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scuba_diving_acronym_questions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return 5 questions from scuba diving "acronym" category."
    
    """
    url = f"https://scuba-diving-quiz-version-101.p.rapidapi.com/trivia/acronym"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scuba-diving-quiz-version-101.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scuba_diving_refresher_questions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return 5 questions and answers from scuba diving "refresher" category."
    
    """
    url = f"https://scuba-diving-quiz-version-101.p.rapidapi.com/trivia/refresher"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scuba-diving-quiz-version-101.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_questions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all scuba diving questions from 3 categories:
		- Refresher
		- Acronym
		- Emergency"
    
    """
    url = f"https://scuba-diving-quiz-version-101.p.rapidapi.com/trivia"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scuba-diving-quiz-version-101.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

