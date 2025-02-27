import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_exercise_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an exercise by it's ID. Valid IDs [0,953]"
    id: Exercise's ID
        
    """
    url = f"https://musclewiki.p.rapidapi.com/exercises/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "musclewiki.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exercises(force: str=None, name: str=None, muscle: str=None, category: str=None, difficulty: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Exercises"
    force: Filter by force
        name: Filter by exercises name
        muscle: Filter by targeted muscle
        category: Filter by category
        difficulty: Filter by difficulty
        
    """
    url = f"https://musclewiki.p.rapidapi.com/exercises"
    querystring = {}
    if force:
        querystring['force'] = force
    if name:
        querystring['name'] = name
    if muscle:
        querystring['muscle'] = muscle
    if category:
        querystring['category'] = category
    if difficulty:
        querystring['difficulty'] = difficulty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "musclewiki.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_attributes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Attributes you can use to filter exercises"
    
    """
    url = f"https://musclewiki.p.rapidapi.com/exercises/attributes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "musclewiki.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

