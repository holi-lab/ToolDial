import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def icecream_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random icecream recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/icecream/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cake_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random cake recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/cake/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pastry_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random pastry recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/pastry/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def side_dish_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random side dish recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/side-dish/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def salad_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random salad recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/salad/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def breakfast_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random breakfast  recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/breakfast/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dinner_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random dinner  recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/dinner/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mediterranean_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random mediterranean  recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/mediterranean/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def appetizer_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random appetizer  recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/appetizer/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lunch_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random lunch  recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/lunch/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def diet_ingredient(ingredient: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint allows users to retrieve a random diet recipe that contains a specific ingredient. Users can make a GET request to the endpoint with the name of the ingredient as a query parameter, and the API will return a JSON response with the given recipe, including the name, list of ingredients, and instructions."
    
    """
    url = f"https://recipe-finder3.p.rapidapi.com/diet/recipes"
    querystring = {'ingredient': ingredient, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recipe-finder3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

