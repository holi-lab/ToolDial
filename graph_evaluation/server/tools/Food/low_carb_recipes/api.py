import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_recipes(maxnetcarbs: int=5, limit: int=10, maxcalories: int=500, maxpreparetime: int=10, excludeingredients: str='cinnamon', tags: str='keto;dairy-free', name: str='cake', maxaddedsugar: int=0, maxsugar: int=3, maxcooktime: int=20, includeingredients: str='egg;butter', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for recipes that match provided criterions"
    maxnetcarbs: Maximum net carbs (total carbs subtract fiber and sugar alcohols) of 1 serving in grams
        limit: Max number of results
        maxcalories: Maximum calories of 1 serving in KCal
        maxpreparetime: Maximum preparation time in minutes
        excludeingredients: Semicolon separated terms to be excluded in ingredients
        tags: Semicolon separated tags, supported tags are: 
```
15-minute-meals
3-ingredient-meals
5-ingredient-meals
appetizer
beef-free
beverages
breakfast
chicken-free
dairy-free
desserts
egg-free
eggs
fish
fish-free
freezer-friendly
french
gluten-free
good-for-leftovers
grains
high-protein
keto
kid-friendly
lchf
low-carb
lunch
main-dishes
meal-plan-ok
msg
no-cooking-required
one-pot-meals
paleo
pantry-recipes
peanut-free
peanuts
pescatarian
pork-free
quick-easy
relevant-meal--beverages
relevant-meal--breakfast
relevant-meal--desserts
relevant-meal--lunch
relevant-meal--main-dishes
relevant-meal--salads
relevant-meal--sides
relevant-meal--snacks
salads
sheet-pan-dinners
shellfish
shellfish-free
sides
skillet
snacks
soy-free
sulphites
tree-nut-free
vegan
vegetarian
wheat-free
whole-30
```
        name: Search terms to be appeared in the recipe name
        maxaddedsugar: Maximum added sugar of 1 serving in grams
        maxsugar: Maximum sugar of 1 serving in grams
        maxcooktime: Maximum cook time in minutes
        includeingredients: Semicolon separated terms to be included in ingredients
        
    """
    url = f"https://low-carb-recipes.p.rapidapi.com/search"
    querystring = {}
    if maxnetcarbs:
        querystring['maxNetCarbs'] = maxnetcarbs
    if limit:
        querystring['limit'] = limit
    if maxcalories:
        querystring['maxCalories'] = maxcalories
    if maxpreparetime:
        querystring['maxPrepareTime'] = maxpreparetime
    if excludeingredients:
        querystring['excludeIngredients'] = excludeingredients
    if tags:
        querystring['tags'] = tags
    if name:
        querystring['name'] = name
    if maxaddedsugar:
        querystring['maxAddedSugar'] = maxaddedsugar
    if maxsugar:
        querystring['maxSugar'] = maxsugar
    if maxcooktime:
        querystring['maxCookTime'] = maxcooktime
    if includeingredients:
        querystring['includeIngredients'] = includeingredients
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "low-carb-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_random_recipe(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random recipe"
    
    """
    url = f"https://low-carb-recipes.p.rapidapi.com/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "low-carb-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recipe_by_id(recipeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of the specified recipe"
    recipeid: Recipe ID
        
    """
    url = f"https://low-carb-recipes.p.rapidapi.com/recipes/{recipeid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "low-carb-recipes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

