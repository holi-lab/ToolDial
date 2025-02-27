import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def diet_plan(unliked_foods: str, allergies: str, height: int, weight: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "meal planner end point which returns custom meal plan in HTML format"
    unliked_foods: (Please NOTE!, if no allergies or unliked foods want to passed, make sure to pass \"\" instead of \"none\", \"N/A\", etc)
        allergies: (Please NOTE!, if no allergies or unliked foods want to passed, make sure to pass \"\" instead of \"none\", \"N/A\", etc)
        
    """
    url = f"https://customized-meal-plan-generator.p.rapidapi.com/diet_plan"
    querystring = {'unliked_foods': unliked_foods, 'allergies': allergies, 'height': height, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "customized-meal-plan-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

