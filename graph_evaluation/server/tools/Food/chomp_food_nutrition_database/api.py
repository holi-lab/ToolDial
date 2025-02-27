import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def advanced_product_search(category: str=None, nutrition: str=None, lifestyle: str=None, keyword: str=None, trace: str=None, palm_oil: str=None, allergen: str=None, mineral: str=None, vitamin: str=None, country: str=None, brand: str=None, ingredient: str='milk', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of products using any combination of our search query parameters. Retrieve barcodes, nutrition labels, ingredients, allergen warnings, trace ingredients, compatible health diets, brands, countries, categories, keywords, minerals, and vitamins for a bulk list of products."
    category: Category that the product falls under. Try "Pasta Dishes"
        nutrition: Item from the nutrition label. Try "Zinc"
        lifestyle: Get products that are compatible with a specific lifestyle (vegan, vegetarian, gluten free). Try "Gluten Free"
        keyword: Keyword used to describe the product. Try "Organic"
        trace: Trace ingredient that may be found in a product. Try "Peanuts"
        palm_oil: Ingredient made from palm oil. Try "E160a Beta Carotene"
        allergen: Ingredient that may cause an allergic reaction. Try "Peanuts"
        mineral: Mineral found in a product. Try "Iron"
        vitamin: Vitamin found in a product. Try "Vitamin B12"
        country: Country of origin. Try "United States"
        brand: The brand that makes the product. Try "Nestle"
        ingredient: Ingredient found in a product. Try "Milk"
        
    """
    url = f"https://chomp.p.rapidapi.com/request.php"
    querystring = {}
    if category:
        querystring['category'] = category
    if nutrition:
        querystring['nutrition'] = nutrition
    if lifestyle:
        querystring['lifestyle'] = lifestyle
    if keyword:
        querystring['keyword'] = keyword
    if trace:
        querystring['trace'] = trace
    if palm_oil:
        querystring['palm_oil'] = palm_oil
    if allergen:
        querystring['allergen'] = allergen
    if mineral:
        querystring['mineral'] = mineral
    if vitamin:
        querystring['vitamin'] = vitamin
    if country:
        querystring['country'] = country
    if brand:
        querystring['brand'] = brand
    if ingredient:
        querystring['ingredient'] = ingredient
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_nutrition_label_titles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of individual items from nutrition labels along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/nutrition.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of countries along with the number of products in our database for each one."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/country.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def individual_product_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an individual product using our internal product ID. Use this endpoint to retrieve barcodes, nutrition labels, ingredients, allergen warnings, trace ingredients, compatible health diets, brands, countries, categories, keywords, minerals, and vitamins for a specific product."
    id: The ID of a product. Try "15"
        
    """
    url = f"https://chomp.p.rapidapi.com/product.php"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_palm_oil_ingredients(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of ingredients made from palm oil along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/palm-oil.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_allergens(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of ingredients that may cause allergic reactions along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/allergen.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of product categories along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/category.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lifestyle_health_diets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of lifestyles/health diets along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/lifestyle.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_minerals(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of minerals along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/mineral.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vitamins(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of product vitamins along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/vitamin.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trace_ingredients(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of trace ingredients along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/trace.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_keywords(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of product keywords along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/keyword.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_brands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of brands along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/brand.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def individual_product_by_barcode(code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an individual product using their barcode. Use this endpoint to retrieve barcodes, nutrition labels, ingredients, allergen warnings, trace ingredients, compatible health diets, brands, countries, categories, keywords, minerals, and vitamins for individual products."
    code: Barcode for the product. Try "0842234000988"
        
    """
    url = f"https://chomp.p.rapidapi.com/product-code.php"
    querystring = {'code': code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ingredients(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of ingredients along with their total product count."
    
    """
    url = f"https://chomp.p.rapidapi.com/data/pull/ingredient.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_search_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for products by name. Use this endpoint to retrieve barcodes, nutrition labels, ingredients, allergen warnings, trace ingredients, compatible health diets, brands, countries, categories, keywords, minerals, and vitamins for individual products."
    name: Name of the product you want to retrieve. Try "Starburst"
        
    """
    url = f"https://chomp.p.rapidapi.com/product-search.php"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chomp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

