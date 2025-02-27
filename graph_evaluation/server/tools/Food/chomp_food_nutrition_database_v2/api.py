import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def food_branded_barcode_php(code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data for a branded food using the food's UPC/EAN barcode."
    code: UPC/EAN barcode
        
    """
    url = f"https://chomp-food-nutrition-database-v2.p.rapidapi.com/food/branded/barcode.php"
    querystring = {'code': code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp-food-nutrition-database-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def food_branded_name_php(name: str, limit: int=3, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for branded food items by name."
    name: Search for branded food items using a general food name keyword. This does not have to exactly match the "official" name for the food.
        limit: Set maximum number of records you want the API to return. The default value is "10."
        page: This is how you paginate the search result. By default, you will see the first 10 records. You must increment the page number to access the next 10 records, and so on. The default value is "1."
        
    """
    url = f"https://chomp-food-nutrition-database-v2.p.rapidapi.com/food/branded/name.php"
    querystring = {'name': name, }
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp-food-nutrition-database-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def food_branded_search_php(page: int=1, country: str='United States', palm_oil: str=None, mineral: str=None, trace: str=None, nutrient: str=None, diet: str=None, vitamin: str=None, brand: str='Starbucks', keyword: str=None, ingredient: str=None, allergen: str=None, limit: str=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for branded food items using various parameters."
    page: This is how you paginate the search result. By default, you will see the first 10 records. You must increment the page number to access the next 10 records, and so on. The default value is "1."
        country: Filter the search to only include branded foods that are sold in a specific country.
        palm_oil: Filter the search to only include branded foods that contain a specific ingredient from palm oil.
        mineral: Filter the search to only include branded foods that contain a specific mineral.
        trace: Filter the search to only include branded foods that contain a specific trace ingredient.
        nutrient: Filter the search to only include branded foods that contain a specific nutrient.
        diet: Filter the search to only include branded foods that are considered compatible with a specific diet. Valid diets: Vegan, Vegetarian, Gluten Free
        vitamin: Filter the search to only include branded foods that contain a specific vitamin.
        brand: Filter the search to only include branded foods that are owned by a specific brand.
        keyword: Filter the search to only include branded foods that contain a specific keyword.
        ingredient: Filter the search to only include branded foods that contain a specific ingredient.
        allergen: Filter the search to only include branded foods that contain a specific allergen.
        limit: Set maximum number of records you want the API to return. The default value is 10.
        category: Filter the search to only include branded foods from a specific category.
        
    """
    url = f"https://chomp-food-nutrition-database-v2.p.rapidapi.com/food/branded/search.php"
    querystring = {}
    if page:
        querystring['page'] = page
    if country:
        querystring['country'] = country
    if palm_oil:
        querystring['palm_oil'] = palm_oil
    if mineral:
        querystring['mineral'] = mineral
    if trace:
        querystring['trace'] = trace
    if nutrient:
        querystring['nutrient'] = nutrient
    if diet:
        querystring['diet'] = diet
    if vitamin:
        querystring['vitamin'] = vitamin
    if brand:
        querystring['brand'] = brand
    if keyword:
        querystring['keyword'] = keyword
    if ingredient:
        querystring['ingredient'] = ingredient
    if allergen:
        querystring['allergen'] = allergen
    if limit:
        querystring['limit'] = limit
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp-food-nutrition-database-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def food_ingredient_search_php(find: str, limit: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data for a specific ingredient or a specific set of ingredients."
    find: Search our database for a single ingredient or a specific set of ingredients. Example #1: Single Ingredient "&find=raw broccoli". Example #2: Set of Ingredients "&find=raw broccoli,raw cauliflower,mashed potatoes". Comma-separated lists cannot contain more than 10 ingredients. You must perform additional API calls if you are looking up more than 10 ingredients.
        limit: Set maximum number of records you want the API to return, per search term. The default value is "1."
        
    """
    url = f"https://chomp-food-nutrition-database-v2.p.rapidapi.com/food/ingredient/search.php"
    querystring = {'find': find, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp-food-nutrition-database-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

