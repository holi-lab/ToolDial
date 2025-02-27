import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dashboard(login_id: str, api_key: str, survey_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the deployment and response dashboard"
    
    """
    url = f"https://community-survey-methods.p.rapidapi.com/{login_id}/{api_key}/responses/{survey_code}/dashboard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-survey-methods.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deployment_details(login_id: str, api_key: str, survey_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API call, you can retrieve the current deployment status of a particular survey."
    
    """
    url = f"https://community-survey-methods.p.rapidapi.com/{login_id}/{api_key}/surveys/{survey_code}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-survey-methods.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def responses_summary(login_id: str, api_key: str, survey_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API call you can retrieve information such as custom field labels & values, date on which  a response was started and when was it completed, the survey response method (email/web), the  email address of the respondent in case of an email response, IP address and the response code  for each individual response of your survey."
    
    """
    url = f"https://community-survey-methods.p.rapidapi.com/{login_id}/{api_key}/responses/{survey_code}/summary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-survey-methods.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_survey_details(login_id: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-survey-methods.p.rapidapi.com/{login_id}/{api_key}/surveys/details"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-survey-methods.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

