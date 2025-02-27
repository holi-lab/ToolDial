import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def filter_protein(protein_in_grams_lt: int, protein_in_grams_gt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives keto recipes within given range of protein value.
		protein_in_grams__lt = Less than
		protein_in_grams__gt = Greater than"
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'protein_in_grams__lt': protein_in_grams_lt, 'protein_in_grams__gt': protein_in_grams_gt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter_carbohydrates(carbohydrates_in_grams_lt: int, carbohydrates_in_grams_gt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives keto recipes within given range of carbohydrates value.
		carbohydrates_in_grams__lt = Less than
		carbohydrates_in_grams__gt = Greater than"
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'carbohydrates_in_grams__lt': carbohydrates_in_grams_lt, 'carbohydrates_in_grams__gt': carbohydrates_in_grams_gt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter_fats(fat_in_grams_lt: int, fat_in_grams_gt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives keto recipes within given range of fat content.
		fat_in_grams__lt = Less than
		fat_in_grams__gt = Greater than"
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'fat_in_grams__lt': fat_in_grams_lt, 'fat_in_grams__gt': fat_in_grams_gt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter_calories(calories_lt: int, calories_gt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives keto recipes within given range of calories.
		calories__lt = Less than
		calories__gt = Greater than"
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'calories__lt': calories_lt, 'calories__gt': calories_gt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter_cook_time_in_minutes(cook_time_in_minutes_tg: int, cook_time_in_minutes_lt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives keto recipes within given range of cooking time.
		cook_time_in_minutes__lt = Less than
		cook_time_in_minutes__gt = Greater than"
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'cook_time_in_minutes__tg': cook_time_in_minutes_tg, 'cook_time_in_minutes__lt': cook_time_in_minutes_lt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filter_prep_time_in_minutes(prep_time_in_minutes_gt: int, prep_time_in_minutes_lt: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives keto recipes within given range of preparation time.
		prep_time_in_minutes__lt = Less than
		prep_time_in_minutes__gt = Greater than"
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'prep_time_in_minutes__gt': prep_time_in_minutes_gt, 'prep_time_in_minutes__lt': prep_time_in_minutes_lt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keto_recipes_by_difficulty(difficulty: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives all keto recipes by its difficulty (Easy, Medium or Difficult)."
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'difficulty': difficulty, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_keto_recipes_by_category(category: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives all keto recipes by category ID."
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all available categories of keto recipe."
    
    """
    url = f"https://keto-diet.p.rapidapi.com/categories/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_keto_recipe(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives all recipes by the search term."
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_keto_recipe(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives a single recipe by ID"
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_keto_recipes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all available keto recipes."
    
    """
    url = f"https://keto-diet.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keto-diet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

