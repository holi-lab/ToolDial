import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_user(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides you all the information of a specific user. This will be useful to present to the user specific information. To update any of those values, you need to use ```PUT /user```"
    
    """
    url = f"https://4eat.p.rapidapi.com/user"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_getdislikes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the list of ingredients that the user can indicate as disliked. We'll remove these from his diet"
    
    """
    url = f"https://4eat.p.rapidapi.com/dislikes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_getrecipes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will provide you a list with all the recipes that you have available as a customer. This includes our recipes, if enabled, and the recipes that you can create. Consider that this method should not be used for user diet selections, as we have a specific endpoint for those cases, where we optimize which recipes can be selected in a meal.
		
		This is a list that you can use to review and check your recipes and analyze the available recipes, but not as a meal selection tool."
    
    """
    url = f"https://4eat.p.rapidapi.com/recipes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_getallergies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the list of allergies and ingredients that the user can indicate as allergy. We'll remove these from his diet"
    
    """
    url = f"https://4eat.p.rapidapi.com/allergies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_getcustomersetup(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is a generic function that provides all the relevant information for you as a customer to setup your environment. As a response, we provide the list of countries, languages, subcategories of ingredients, ingredients, subcategories of recipes, allergies, etc.
		
		This is a critical endpoint that you will need to define several properties and to get some important data as ingredients, in order to define your own recipes."
    
    """
    url = f"https://4eat.p.rapidapi.com/setupData"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_getcreaterecipeconstraints(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the constraints for creating an user-specific recipe for a certain meal."
    
    """
    url = f"https://4eat.p.rapidapi.com/userRecipe"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_getrecipeimage(recipeid: int, thumb: str='thumb', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides you the image associated to the requested recipe from 4eat. As your recipes are defined by yourself, and only you have access to them, you will be in charge of the images for your own recipes. If you decide to use our recipes, this endpoint gives you those images."
    thumb: Add ```/thumb``` to endpoint to get recipe thumbnail image
        
    """
    url = f"https://4eat.p.rapidapi.com/recipeImage/{recipeid}/{thumb}"
    querystring = {}
    if thumb:
        querystring['thumb'] = thumb
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_getshoppinglist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the list of ingredients and their quantities for a range of weeks, days or meals. You can select as many ranges as you need, to get the specific list of ingredients that you need.
		
		For every range, you can define:
		- Week
		- Week and day
		- Week, day and meal
		
		We also convert the quantities to a measurement that makes sense depending on the ingredient (grams, units, slices, etc)"
    
    """
    url = f"https://4eat.p.rapidapi.com/shoppingList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recipeformeal(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method provides your users the list of available recipes for a meal.
		
		Available means that they meet the nutritional requirements for the user in that specific moment, considering the rest of the meals already selected. Our algorithms ensure that the alternatives that your users can select at each meal are fitted to their requirements in terms of amounts, allergies, likes and dislikes, macronutrients and ingredients balance.
		
		This can be used to select a blank meal or to replace an existing recipe on that meal."
    
    """
    url = f"https://4eat.p.rapidapi.com/recipeForMeal"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_getweek(weekid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method returns a previously generated week for a user."
    
    """
    url = f"https://4eat.p.rapidapi.com/week/{weekid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4eat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

