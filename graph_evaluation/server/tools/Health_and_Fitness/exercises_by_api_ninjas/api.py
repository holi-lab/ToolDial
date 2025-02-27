import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_exercises(offset: int=None, muscle: str='biceps', type: str=None, difficulty: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Exercises API endpoint. Returns up to 10 exercises that satisfy the given parameters."
    offset: number of results to offset for pagination. Default is 0.
        muscle: muscle group targeted by the exercise. Possible values are:

abdominals
abductors
adductors
biceps
calves
chest
forearms
glutes
hamstrings
lats
lower_back
middle_back
neck
quadriceps
traps
triceps
        type: exercise type. Possible values are:

cardio
olympic_weightlifting
plyometrics
powerlifting
strength
stretching
strongman
        difficulty: difficulty level of the exercise. Possible values are:

beginner
intermediate
expert
        name: name of exercise. This value can be partial (e.g. press will match Dumbbell Bench Press)
        
    """
    url = f"https://exercises-by-api-ninjas.p.rapidapi.com/v1/exercises"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if muscle:
        querystring['muscle'] = muscle
    if type:
        querystring['type'] = type
    if difficulty:
        querystring['difficulty'] = difficulty
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exercises-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

