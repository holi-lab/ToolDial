import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def workout_plan_create(goals: str='chest', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The end point takes a GET request with string(e.g."Chest","Triceps","Back","Biceps","Shoulder","Legs","Abs","Cardio") as Parameter and return workout plan.
		Example1:https://vineetyyadav.pythonanywhere.com/workout-plan?goals=abs
		Example2:https://vineetyyadav.pythonanywhere.com/workout-plan?goals=chest"
    
    """
    url = f"https://workout-plan-creator.p.rapidapi.com/workout-plan"
    querystring = {}
    if goals:
        querystring['goals'] = goals
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "workout-plan-creator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

