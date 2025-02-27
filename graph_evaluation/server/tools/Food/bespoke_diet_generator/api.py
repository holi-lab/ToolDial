import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_food_replacement_options_in_diet(dayindex: int, ingredientid: str, mealtype: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the list of ingredients that can replace another ingredient in an existing diet."
    dayindex: The index of the day for a given diet
        ingredientid: The id of the ingredient that will be changed.
        mealtype: The meal type for which the change should be made.
        userid: Existing user.
        
    """
    url = f"https://bespoke-diet-generator.p.rapidapi.com/user/{userid}/diet/{dayindex}/{mealtype}/{ingredientid}/replace"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bespoke-diet-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_information(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the information of the user with the matching user ID."
    userid: Id of an existing user.
        
    """
    url = f"https://bespoke-diet-generator.p.rapidapi.com/user/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bespoke-diet-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_user_s_food_preferences(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the list of all ingredients that will be excluded from user's diets."
    userid: The user that does not want these ingredients in their diet.
        
    """
    url = f"https://bespoke-diet-generator.p.rapidapi.com/user/{userid}/ingredients/excluded"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bespoke-diet-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_diet(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the diet configuration and the diet plan for each day."
    userid: The user for which the diet will be generated.
        
    """
    url = f"https://bespoke-diet-generator.p.rapidapi.com/user/{userid}/diet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bespoke-diet-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_diet_for_a_specific_day(dayindex: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current diet plan for the specified user for a specific day."
    dayindex: The day index of the diet.
        userid: The user id.
        
    """
    url = f"https://bespoke-diet-generator.p.rapidapi.com/user/{userid}/diet/{dayindex}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bespoke-diet-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ingredients(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all the ingredients."
    
    """
    url = f"https://bespoke-diet-generator.p.rapidapi.com/ingredients"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bespoke-diet-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

