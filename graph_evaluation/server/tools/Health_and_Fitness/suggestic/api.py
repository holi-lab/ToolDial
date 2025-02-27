import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_user(authorization: str='Bearer', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each Suggestic partner/client has the ability to create their own set of users."
    
    """
    url = f"https://suggestic1.p.rapidapi.com/"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "suggestic1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def remove_a_meal_plan(authorization: str='Bearer', auth: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the **removeMealPlan** mutation to remove the current and following days in the profile's meal plan. Therefore, it is not possible to delete meal plans which dates have already passed."
    
    """
    url = f"https://suggestic1.p.rapidapi.com/"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    if auth:
        querystring['auth'] = auth
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "suggestic1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

