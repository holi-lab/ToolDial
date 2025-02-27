import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def enneagram_personality_test_questionnaire(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This route will give you all questions for Enneagram Personality Test."
    
    """
    url = f"https://personality-quest.p.rapidapi.com/enneagram_personality_test/start_guest_test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personality-quest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def big_five_personality_test(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This route is for getting Big Five Personality Test Questionnaire"
    
    """
    url = f"https://personality-quest.p.rapidapi.com/big_five_personality_test/start_guest_test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personality-quest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iq_personality_test_questionnaire(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This route will give you all questions for IQ Personality Test."
    
    """
    url = f"https://personality-quest.p.rapidapi.com/iq_test/start_guest_test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personality-quest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def emotional_intelligence_personality_test_questionnaire(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This route will give you all questions for Emotional Intelligence  Personality Test."
    
    """
    url = f"https://personality-quest.p.rapidapi.com/emotional_intelligence_test/start_guest_test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personality-quest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def career_aptitude_personality_test_questionnaire(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This route will give you all questions for Career Aptitude Personality Test."
    
    """
    url = f"https://personality-quest.p.rapidapi.com/career_personality_test/start_guest_test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personality-quest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pq_personality_test_questionnaire(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This route will give you all questions for Positive Quotient    Personality Test."
    
    """
    url = f"https://personality-quest.p.rapidapi.com/positive_quotient_test/start"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personality-quest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def disc_personality_test_questionnaire(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This route will give you all questions for DISC Personality Test."
    
    """
    url = f"https://personality-quest.p.rapidapi.com/disc/start_guest_test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personality-quest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sixteen_type_personality_questionnaire(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This route is for getting Sixteen Type Personality Test Questionnaire."
    
    """
    url = f"https://personality-quest.p.rapidapi.com/sixteen_personality_test/start_guest_test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personality-quest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

