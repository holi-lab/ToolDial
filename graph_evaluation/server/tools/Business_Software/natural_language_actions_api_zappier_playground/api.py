import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_configuration_link(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a link to configure more actions. Alternatively, searching through apps and actions will provide more specific configuration links."
    
    """
    url = f"https://natural-language-actions-api-zappier-playground.p.rapidapi.com/api/v1/configuration-link/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "natural-language-actions-api-zappier-playground.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test that the API and auth are working."
    
    """
    url = f"https://natural-language-actions-api-zappier-playground.p.rapidapi.com/api/v1/check/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "natural-language-actions-api-zappier-playground.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_execution_log_endpoint(execution_log_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the execution log for a given execution log id."
    
    """
    url = f"https://natural-language-actions-api-zappier-playground.p.rapidapi.com/api/v1/execution-log/{execution_log_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "natural-language-actions-api-zappier-playground.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_exposed_actions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all the currently exposed actions for the given account."
    
    """
    url = f"https://natural-language-actions-api-zappier-playground.p.rapidapi.com/api/v1/exposed/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "natural-language-actions-api-zappier-playground.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

