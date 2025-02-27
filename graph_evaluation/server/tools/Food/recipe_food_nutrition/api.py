import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recipe_nutrition_by_id_image(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Visualize a recipe's nutritional information as an image."
    id: The recipe id.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/nutritionWidget.png"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipe_nutrition_label_image(is_id: int, showingredients: bool=None, showzerovalues: bool=None, showoptionalnutrients: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a recipe's nutrition label as an image."
    id: The recipe id.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/nutritionLabel.png"
    querystring = {}
    if showingredients:
        querystring['showIngredients'] = showingredients
    if showzerovalues:
        querystring['showZeroValues'] = showzerovalues
    if showoptionalnutrients:
        querystring['showOptionalNutrients'] = showoptionalnutrients
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipe_nutrition_by_id_widget(is_id: str, defaultcss: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Visualize a recipe's nutritional information as HTML including CSS."
    id: The recipe id.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/nutritionWidget"
    querystring = {}
    if defaultcss:
        querystring['defaultCss'] = defaultcss
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_all_food(query: str, offset: str='0', number: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search all food content with one call. That includes recipes, grocery products, menu items, simple foods (ingredients), and food videos."
    
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/search"
    querystring = {'query': query, }
    if offset:
        querystring['offset'] = offset
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_meal_plan_day(username: str, date: str, hash: str='4b5v4398573406', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a meal planned day for the given user. The username must be a spoonacular user and the hash must the the user's hash that can be found in his/her account."
    username: The username.
        date: The date of the meal planned day in the format yyyy-mm-dd.
        hash: The private hash for the username.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/mealplanner/{username}/day/{date}"
    querystring = {}
    if hash:
        querystring['hash'] = hash
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compute_ingredient_amount(nutrient: str, target: int, is_id: str, unit: str='oz', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compute the amount you need of a certain ingredient for a certain nutritional goal. For example, how much soy milk do you have to drink to get 10 grams of protein?"
    target: The target number of the given nutrient.
        id: The id of the ingredient you want the amount for. You can use the Ingredient Search endpoint to get the ingredient id.
        unit: The target unit.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/ingredients/{is_id}/amount"
    querystring = {'nutrient': nutrient, 'target': target, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ingredient_search(query: str, mincarbspercent: int=5, sort: str='calories', intolerances: str='egg', minfatpercent: int=1, maxproteinpercent: int=50, number: int=10, maxcarbspercent: int=30, sortdirection: str='asc', offset: str='0', addchildren: bool=None, minproteinpercent: int=5, maxfatpercent: int=10, metainformation: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for simple whole foods (e.g. fruits, vegetables, nuts, grains, meat, fish, dairy etc.)."
    query: The partial or full ingredient name.
        mincarbspercent: The minimum percentage of carbs the food must have (between 0 and 100).
        sort: The strategy to sort recipes by. [See a full list of supported sorting options.](https://spoonacular.com/food-api/docs#Recipe-Sorting-Options)
        intolerances: A comma-separated list of intolerances. All recipes returned must not contain ingredients that are not suitable for people with the intolerances entered. [See a full list of supported intolerances.](https://spoonacular.com/food-api/docs#Intolerances)
        minfatpercent: The minimum percentage of fat the food must have (between 0 and 100).
        maxproteinpercent: The maximum percentage of protein the food can have (between 0 and 100).
        number: The number of expected results (between 1 and 100).
        maxcarbspercent: The maximum percentage of carbs the food can have (between 0 and 100).
        sortdirection: The direction in which to sort. Must be either 'asc' (ascending) or 'desc' (descending).
        offset: The number of results to skip (between 0 and 990).
        addchildren: Whether to add children of found foods.
        minproteinpercent: The minimum percentage of protein the food must have (between 0 and 100).
        maxfatpercent: The maximum percentage of fat the food can have (between 0 and 100).
        metainformation: Whether to return more meta information about the ingredients.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/ingredients/search"
    querystring = {'query': query, }
    if mincarbspercent:
        querystring['minCarbsPercent'] = mincarbspercent
    if sort:
        querystring['sort'] = sort
    if intolerances:
        querystring['intolerances'] = intolerances
    if minfatpercent:
        querystring['minFatPercent'] = minfatpercent
    if maxproteinpercent:
        querystring['maxProteinPercent'] = maxproteinpercent
    if number:
        querystring['number'] = number
    if maxcarbspercent:
        querystring['maxCarbsPercent'] = maxcarbspercent
    if sortdirection:
        querystring['sortDirection'] = sortdirection
    if offset:
        querystring['offset'] = offset
    if addchildren:
        querystring['addChildren'] = addchildren
    if minproteinpercent:
        querystring['minProteinPercent'] = minproteinpercent
    if maxfatpercent:
        querystring['maxFatPercent'] = maxfatpercent
    if metainformation:
        querystring['metaInformation'] = metainformation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nutrition_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a recipe's nutrition widget data."
    id: The recipe id.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/nutritionWidget.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equipment_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a recipe's equipment list."
    id: The recipe id.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/equipmentWidget.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_meal_plan_week(start_date: str, username: str, hash: str='4b5v4398573406', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a meal planned week for the given user. The username must be a spoonacular user and the hash must the the user's hash that can be found in his/her account."
    start_date: The start date of the meal planned week in the format yyyy-mm-dd.
        username: The username.
        hash: The private hash for the username.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/mealplanner/{username}/week/{start_date}"
    querystring = {}
    if hash:
        querystring['hash'] = hash
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversation_suggests(query: str, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns suggestions for things the user can say or ask the chat bot."
    query: A (partial) query from the user. The endpoint will return if it matches topics it can talk about.
        number: The number of suggestions to return must be in interval [1,25].
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/converse/suggest"
    querystring = {'query': query, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wine_recommendation(wine: str, maxprice: int=50, number: int=3, minrating: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific wine recommendation (concrete product) for a given wine type, e.g. "merlot"."
    wine: The name of the wine to get a specific product recommendation for.
        maxprice: The maximum price of the recommended wine.
        number: The number of wine recommendations expected.
        minrating: The minimum rating of the recommended wine between 0 and 1. For example, 0.8 equals 4 out of 5 stars.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/wine/recommendation"
    querystring = {'wine': wine, }
    if maxprice:
        querystring['maxPrice'] = maxprice
    if number:
        querystring['number'] = number
    if minrating:
        querystring['minRating'] = minrating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wine_description(wine: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a simple description of a certain wine, e.g. "malbec", "riesling", or "merlot"."
    wine: The name of the wine, e.g. \"malbec\", \"riesling\", or \"merlot\".
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/wine/description"
    querystring = {'wine': wine, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wine_pairing(food: str, maxprice: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a wine that goes well with a food. Food can be a dish name ("lasagna"), an ingredient name ("salmon"), or a cuisine ("italian")."
    food: The food to get a pairing for. This can be a dish (\"steak\"), an ingredient (\"salmon\"), or a cuisine (\"italian\").
        maxprice: The maximum price for the specific wine recommendation.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/wine/pairing"
    querystring = {'food': food, }
    if maxprice:
        querystring['maxPrice'] = maxprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_menu_item_information(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a certain menu item."
    
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/menuItems/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_information(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a packaged food product."
    id: The id of the packaged food product.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/products/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_recipes(tags: str='vegetarian,dessert', number: int=1, limitlicense: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find random (popular) recipes."
    tags: Tags that the random recipe(s) must adhere to.
        number: The number of random recipes to be returned. Must be in interval [1,100]. NOTE: Each random recipe returned counts as one request.
        limitlicense: Whether the recipes should have an open license that allows for displaying with proper attribution.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/random"
    querystring = {}
    if tags:
        querystring['tags'] = tags
    if number:
        querystring['number'] = number
    if limitlicense:
        querystring['limitLicense'] = limitlicense
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete_ingredient_search(query: str, number: int=10, metainformation: bool=None, intolerances: str='egg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Autocomplete a search for an ingredient."
    query: The query - a partial or full ingredient name.
        number: The number of results to return, between [1,100]
        metainformation: Whether to return more meta information about the ingredients.
        intolerances: A comma-separated list of intolerances. All found ingredients must not cause problems for people with one of the given tolerances. Possible values are: dairy, egg, gluten, peanut, sesame, seafood, shellfish, soy, sulfite, tree nut, and wheat.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/ingredients/autocomplete"
    querystring = {'query': query, }
    if number:
        querystring['number'] = number
    if metainformation:
        querystring['metaInformation'] = metainformation
    if intolerances:
        querystring['intolerances'] = intolerances
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_similar_recipes(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find recipes which are similar to the given one."
    id: The id of the source recipe to which similar recipes should be found.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/similar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_grocery_products_by_upc(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a food product given its UPC."
    
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/products/upc/{upc}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comparable_products(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find comparable products to the given one."
    upc: The UPC code of the product for that you want to find comparable products.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/products/upc/{upc}/comparable"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recipe_nutrition_label_widget(is_id: int, defaultcss: bool=None, showingredients: bool=None, showoptionalnutrients: bool=None, showzerovalues: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a recipe's nutrition label as an HTML widget."
    id: The recipe id.
        defaultcss: Whether the default CSS should be added to the response.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/nutritionLabel"
    querystring = {}
    if defaultcss:
        querystring['defaultCss'] = defaultcss
    if showingredients:
        querystring['showIngredients'] = showingredients
    if showoptionalnutrients:
        querystring['showOptionalNutrients'] = showoptionalnutrients
    if showzerovalues:
        querystring['showZeroValues'] = showzerovalues
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def talk_to_chatbot(text: str, contextid: str='342938', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can be used to have a conversation about food with the spoonacular chat bot. Use the chat suggests endpoint to show your user what he or she can say."
    text: The request/question/answer from the user to the chat bot.
        contextid: An arbitrary globally unique id for your conversation. The conversation can contain states so you should pass your context id if you want the bot to be able to remember the conversation.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/converse"
    querystring = {'text': text, }
    if contextid:
        querystring['contextId'] = contextid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_site_content(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search spoonacular's site content. You'll be able to find everything that you could also find using the search suggests on spoonacular.com. This is a suggest API so you can send partial strings as queries."
    query: The query to search for. You can also use partial queries such as "spagh" to already find spaghetti recipes, articles, grocery products, and other content.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/site/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_analysis_url(imageurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Classify and analyze a food image by a given URL."
    
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/images/analyze"
    querystring = {'imageUrl': imageurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_random_food_joke(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random joke that includes or is about food."
    
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/jokes/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_custom_foods(hash: str, username: str, query: str, offset: int=0, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search custom foods in a user's account."
    hash: The private hash for the username.
        username: The username.
        query: The search query.
        offset: The number of results to skip (between 0 and 990).
        number: The number of expected results (between 1 and 100).
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/customFoods/search"
    querystring = {'hash': hash, 'username': username, 'query': query, }
    if offset:
        querystring['offset'] = offset
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_shopping_list(username: str, hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current shopping list for the given user."
    username: The username.
        hash: The private hash for the username.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/mealplanner/{username}/shopping-list"
    querystring = {'hash': hash, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_meal_plan_template(is_id: int, username: str, hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a meal plan template."
    id: The shopping list item id.
        username: The username.
        hash: The private hash for the username.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/mealplanner/{username}/templates/{is_id}"
    querystring = {'hash': hash, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_meal_plan_templates(hash: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get meal plan templates from user or public ones. This documentation is for getting user templates. You can also get public templates. Read more about this [here](https://spoonacular.com/food-api/docs#Get-Meal-Plan-Templates)."
    hash: The private hash for the username.
        username: The username.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/mealplanner/{username}/templates"
    querystring = {'hash': hash, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_amounts(ingredientname: str, targetunit: str, sourceunit: str='cups', sourceamount: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert amounts like "2 cups of flour to grams"."
    ingredientname: The ingredient which you want to convert, e.g. the flour in "2.5 cups of flour to grams"
        targetunit: The unit to which you want to convert, e.g. the grams in "2.5 cups of flour to grams". You can also use "piece", e.g. "3.4 oz tomatoes to piece"
        sourceunit: The unit from which you want to convert, e.g. the cups in "2.5 cups of flour to grams". You can also use "serving" or "piece".
        sourceamount: The amount from which you want to convert, e.g. the 2.5 in "2.5 cups of flour to grams"
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/convert"
    querystring = {'ingredientName': ingredientname, 'targetUnit': targetunit, }
    if sourceunit:
        querystring['sourceUnit'] = sourceunit
    if sourceamount:
        querystring['sourceAmount'] = sourceamount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analyze_a_recipe_search_query(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parse a recipe search query to find out its intention."
    q: The recipe search query.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/queries/analyze"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_recipe_from_website(url: str, forceextraction: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract recipe data from a recipe blog or Web page."
    url: The URL of the recipe page.
        forceextraction: If true, the extraction will be triggered no matter whether we know the recipe already. Use that only if information is missing as this operation is slower.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/extract"
    querystring = {'url': url, }
    if forceextraction:
        querystring['forceExtraction'] = forceextraction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ingredients_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a recipe's ingredient list."
    id: The recipe id.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/ingredientWidget.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete_recipe_search(query: str, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Autocomplete a partial input to possible recipe names."
    query: The query to be autocompleted.
        number: The number of results between [1,25].
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/autocomplete"
    querystring = {'query': query, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recipe_information_bulk(ids: str, includenutrition: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about multiple recipes at once. That is equivalent of calling the Get Recipe Information endpoint multiple times but is faster. Note that each returned recipe counts as one request."
    ids: A comma-separated list of recipe ids.
        includenutrition: Include nutrition data to the recipe information.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/informationBulk"
    querystring = {'ids': ids, }
    if includenutrition:
        querystring['includeNutrition'] = includenutrition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recipe_information(is_id: int, includenutrition: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a recipe."
    id: The id of the recipe.
        includenutrition: Include nutrition data to the recipe information. Nutrition data is per serving. If you want the nutrition data for the entire recipe, just multiply by the number of servings.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/information"
    querystring = {}
    if includenutrition:
        querystring['includeNutrition'] = includenutrition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_recipes_deprecated(query: str, number: int=10, type: str='main course', diet: str='vegetarian', excludeingredients: str='coconut', limitlicense: bool=None, offset: int=0, instructionsrequired: bool=None, cuisine: str=None, intolerances: str='egg, gluten', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search recipes in natural language."
    query: The (natural language) recipe search query.
        number: The number of results to return (between 0 and 100).
        type: The type of the recipes. One of the following: main course, side dish, dessert, appetizer, salad, bread, breakfast, soup, beverage, sauce, or drink.
        diet: The diet to which the recipes must be compliant. Possible values are: pescetarian, lacto vegetarian, ovo vegetarian, vegan, and vegetarian.
        excludeingredients: An comma-separated list of ingredients or ingredient types that must not be contained in the recipes.
        limitlicense: Whether the recipes should have an open license that allows for displaying with proper attribution.
        offset: The number of results to skip (between 0 and 900).
        instructionsrequired: Whether the recipes must have instructions.
        cuisine: The cuisine(s) of the recipes. One or more (comma separated) of the following: african, chinese, japanese, korean, vietnamese, thai, indian, british, irish, french, italian, mexican, spanish, middle eastern, jewish, american, cajun, southern, greek, german, nordic, eastern european, caribbean, or latin american.
        intolerances: A comma-separated list of intolerances. All found recipes must not have ingredients that could cause problems for people with one of the given tolerances. Possible values are: dairy, egg, gluten, peanut, sesame, seafood, shellfish, soy, sulfite, tree nut, and wheat.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/search"
    querystring = {'query': query, }
    if number:
        querystring['number'] = number
    if type:
        querystring['type'] = type
    if diet:
        querystring['diet'] = diet
    if excludeingredients:
        querystring['excludeIngredients'] = excludeingredients
    if limitlicense:
        querystring['limitLicense'] = limitlicense
    if offset:
        querystring['offset'] = offset
    if instructionsrequired:
        querystring['instructionsRequired'] = instructionsrequired
    if cuisine:
        querystring['cuisine'] = cuisine
    if intolerances:
        querystring['intolerances'] = intolerances
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_recipes_complex_deprecated(number: int, limitlicense: bool, offset: int, miniron: int=0, mincalcium: int=0, maxmagnesium: int=1000, maxvitaminb5: int=1000, maxvitamina: int=5000, addrecipeinformation: bool=None, minzinc: int=0, minfolate: int=0, maxmanganese: int=1000, maxpotassium: int=1000, maxfiber: int=1000, fillingredients: bool=None, minsodium: int=0, maxcopper: int=1000, minvitaminb3: int=0, maxiodine: int=1000, excludeingredients: str='coconut, mango', mincarbs: int=5, maxcaffeine: int=1000, query: str='burger', minvitamine: int=0, maxcalcium: int=1000, minfiber: int=0, maxcalories: int=1500, maxfolate: int=1000, maxcholesterol: int=1000, mincaffeine: int=0, minvitaminb5: int=0, maxcarbs: int=100, maxvitaminc: int=1000, maxphosphorus: int=1000, mincalories: int=150, intolerances: str='peanut, shellfish', minvitaminb2: int=0, minselenium: int=0, minvitamink: int=0, minmagnesium: int=0, maxsodium: int=1000, maxvitaminb2: int=1000, maxsugar: int=1000, maxvitamind: int=1000, instructionsrequired: bool=None, maxfolicacid: int=1000, ranking: int=2, maxprotein: int=100, miniodine: int=0, maxvitaminb3: int=1000, minvitaminb6: int=0, type: str='main course', mincholine: int=0, maxvitamine: int=1000, equipment: str='pan', minvitaminb1: int=0, minvitamina: int=0, maxvitaminb1: int=1000, minvitaminb12: int=0, minpotassium: int=0, maxiron: int=1000, mincopper: int=0, maxfluoride: int=1000, minvitamind: int=0, maxsaturatedfat: int=50, maxselenium: int=1000, maxvitaminb12: int=1000, mincholesterol: int=0, minprotein: int=5, minfolicacid: int=0, maxzinc: int=1000, minfat: int=5, maxcholine: int=1000, minvitaminc: int=0, minfluoride: int=0, maxalcohol: int=1000, author: str=None, minmanganese: int=0, minsugar: int=0, cuisine: str='american', minalcohol: int=0, includeingredients: str='onions, lettuce, tomato', minsaturatedfat: int=0, diet: str=None, maxvitaminb6: int=1000, minphosphorus: int=0, maxvitamink: int=1000, maxfat: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search through hundreds of thousands of recipes using advanced filtering and ranking. NOTE: Since this method combines three other functionalities, each request counts as 3 requests."
    number: The number of results to return (between 1 and 100).
        limitlicense: Whether the recipes should have an open license that allows for displaying with proper attribution.
        offset: The number of results to skip (between 0 and 900).
        miniron: The minimum number of milligrams of iron the recipe must have.
        mincalcium: The minimum number of milligrams of calcium the recipe must have.
        maxmagnesium: The maximum number of milligrams of magnesium the recipe can have.
        maxvitaminb5: The maximum number of milligrams of Vitamin B5 the recipe can have.
        maxvitamina: The maximum number of IU of Vitamin A the recipe can have.
        addrecipeinformation: If set to true, you get more information about the recipes returned. This saves the calls to get recipe information.
        minzinc: The minimum number of milligrams of zinc the recipe must have.
        minfolate: The minimum number of micrograms of folate the recipe must have.
        maxmanganese: The maximum number of milligrams of manganese the recipe can have.
        maxpotassium: The maxnimum number of milligrams of potassium the recipe can have.
        maxfiber: The maximum number of grams of fiber the recipe can have.
        fillingredients: Add information about the used and missing ingredients in each recipe.
        minsodium: The minimum number of milligrams of sodium the recipe must have.
        maxcopper: The maximum number of milligrams of copper the recipe can have.
        minvitaminb3: The minimum number of milligrams of Vitamin B3 the recipe must have.
        maxiodine: The maximum number of micrograms of iodine the recipe can have.
        excludeingredients: An comma-separated list of ingredients that must not be contained in the recipes.
        mincarbs: The minimum number of grams of carbohydrates the recipe must have.
        maxcaffeine: The maximum number of milligrams of caffeine the recipe can have.
        query: The recipe search query.
        minvitamine: The minimum number of milligrams of Vitamin E the recipe must have.
        maxcalcium: The maximum number of milligrams of calcium the recipe can have.
        minfiber: The minimum number of grams of fiber the recipe must have.
        maxcalories: The maximum number of calories the recipe can have.
        maxfolate: The maximum number of micrograms of folate the recipe can have.
        maxcholesterol: The maximum number of milligrams of cholesterol the recipe can have.
        mincaffeine: The minimum number of milligrams of caffeine the recipe must have.
        minvitaminb5: The minimum number of milligrams of Vitamin B5 the recipe must have.
        maxcarbs: The maximum number of grams of carbohydrates the recipe can have.
        maxvitaminc: The maximum number of milligrams of Vitamin C the recipe can have.
        maxphosphorus: The maximum number of milligrams of phosphorus the recipe can have.
        mincalories: The minimum number of calories the recipe must have.
        intolerances: A comma-separated list of intolerances. All found recipes must not have ingredients that could cause problems for people with one of the given tolerances. Possible values are: dairy, egg, gluten, peanut, sesame, seafood, shellfish, soy, sulfite, tree nut, and wheat.
        minvitaminb2: The minimum number of milligrams of Vitamin B2 the recipe must have.
        minselenium: The minimum number of micrograms of selenium the recipe must have.
        minvitamink: The minimum number of micrograms of Vitamin K the recipe must have.
        minmagnesium: The minimum number of milligrams of magnesium the recipe must have.
        maxsodium: The maximum number of milligrams of sodium the recipe can have.
        maxvitaminb2: The maximum number of milligrams of Vitamin B2 the recipe can have.
        maxsugar: The maximum number of grams of sugar the recipe can have.
        maxvitamind: The maximum number of micrograms of Vitamin D the recipe can have.
        instructionsrequired: Whether the recipes must have instructions.
        maxfolicacid: The maximum number of micrograms of folic acid the recipe can have.
        ranking: Whether to minimize missing ingredients (0), maximize used ingredients (1) first, or rank recipes by relevance (2).
        maxprotein: The maximum number of grams of protein the recipe can have.
        miniodine: The minimum number of micrograms of iodine the recipe must have.
        maxvitaminb3: The maximum number of milligrams of Vitamin B3 the recipe can have.
        minvitaminb6: The minimum number of milligrams of Vitamin B6 the recipe must have.
        type: The type of the recipes. One of the following: main course, side dish, dessert, appetizer, salad, bread, breakfast, soup, beverage, sauce, or drink.
        mincholine: The minimum number of milligrams of choline the recipe must have.
        maxvitamine: The maximum number of milligrams of Vitamin E the recipe can have.
        equipment: The equipment required. Multiple values will be interpreted as 'or'. For example, value could be \"blender, frying pan, bowl\"
        minvitaminb1: The minimum number of milligrams of Vitamin B1 the recipe must have.
        minvitamina: The minimum number of IU of Vitamin A the recipe must have.
        maxvitaminb1: The maximum number of milligrams of Vitamin B1 the recipe can have.
        minvitaminb12: The minimum number of micrograms of Vitamin B12 the recipe must have.
        minpotassium: The minimum number of milligrams of potassium the recipe must have.
        maxiron: The maximum number of milligrams of iron the recipe can have.
        mincopper: The minimum number of milligrams of copper the recipe must have.
        maxfluoride: The maximum number of milligrams of fluoride the recipe can have.
        minvitamind: The minimum number of micrograms of Vitamin D the recipe must have.
        maxsaturatedfat: The maximum number of grams of saturated fat the recipe can have.
        maxselenium: The maximum number of micrograms of selenium the recipe can have.
        maxvitaminb12: The maximum number of micrograms of Vitamin B12 the recipe can have.
        mincholesterol: The minimum number of milligrams of cholesterol the recipe must have.
        minprotein: The minimum number of grams of protein the recipe must have.
        minfolicacid: The minimum number of micrograms of folic acid the recipe must have.
        maxzinc: The maximum number of milligrams of zinc the recipe can have.
        minfat: The minimum number of grams of fat the recipe must have.
        maxcholine: The maximum number of milligrams of choline the recipe can have.
        minvitaminc: The minimum number of milligrams of Vitamin C the recipe must have.
        minfluoride: The minimum number of milligrams of fluoride the recipe must have.
        maxalcohol: The maximum number of grams of alcohol the recipe can have.
        author: The username of the recipe author.
        minmanganese: The minimum number of milligrams of manganese the recipe must have.
        minsugar: The minimum number of grams of sugar the recipe must have.
        cuisine: The cuisine(s) of the recipes. One or more (comma separated) of the following: african, chinese, japanese, korean, vietnamese, thai, indian, british, irish, french, italian, mexican, spanish, middle eastern, jewish, american, cajun, southern, greek, german, nordic, eastern european, caribbean, or latin american.
        minalcohol: The minimum number of grams of alcohol the recipe must have.
        includeingredients: A comma-separated list of ingredients that should/must be contained in the recipe.
        minsaturatedfat: The minimum number of grams of saturated fat the recipe must have.
        diet: The diet to which the recipes must be compliant. Possible values are: pescetarian, lacto vegetarian, ovo vegetarian, vegan, paleo, primal, and vegetarian.
        maxvitaminb6: The maximum number of milligrams of Vitamin B6 the recipe can have.
        minphosphorus: The minimum number of milligrams of phosphorus the recipe must have.
        maxvitamink: The maximum number of micrograms of Vitamin K the recipe can have.
        maxfat: The maximum number of grams of fat the recipe can have.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/searchComplex"
    querystring = {'number': number, 'limitLicense': limitlicense, 'offset': offset, }
    if miniron:
        querystring['minIron'] = miniron
    if mincalcium:
        querystring['minCalcium'] = mincalcium
    if maxmagnesium:
        querystring['maxMagnesium'] = maxmagnesium
    if maxvitaminb5:
        querystring['maxVitaminB5'] = maxvitaminb5
    if maxvitamina:
        querystring['maxVitaminA'] = maxvitamina
    if addrecipeinformation:
        querystring['addRecipeInformation'] = addrecipeinformation
    if minzinc:
        querystring['minZinc'] = minzinc
    if minfolate:
        querystring['minFolate'] = minfolate
    if maxmanganese:
        querystring['maxManganese'] = maxmanganese
    if maxpotassium:
        querystring['maxPotassium'] = maxpotassium
    if maxfiber:
        querystring['maxFiber'] = maxfiber
    if fillingredients:
        querystring['fillIngredients'] = fillingredients
    if minsodium:
        querystring['minSodium'] = minsodium
    if maxcopper:
        querystring['maxCopper'] = maxcopper
    if minvitaminb3:
        querystring['minVitaminB3'] = minvitaminb3
    if maxiodine:
        querystring['maxIodine'] = maxiodine
    if excludeingredients:
        querystring['excludeIngredients'] = excludeingredients
    if mincarbs:
        querystring['minCarbs'] = mincarbs
    if maxcaffeine:
        querystring['maxCaffeine'] = maxcaffeine
    if query:
        querystring['query'] = query
    if minvitamine:
        querystring['minVitaminE'] = minvitamine
    if maxcalcium:
        querystring['maxCalcium'] = maxcalcium
    if minfiber:
        querystring['minFiber'] = minfiber
    if maxcalories:
        querystring['maxCalories'] = maxcalories
    if maxfolate:
        querystring['maxFolate'] = maxfolate
    if maxcholesterol:
        querystring['maxCholesterol'] = maxcholesterol
    if mincaffeine:
        querystring['minCaffeine'] = mincaffeine
    if minvitaminb5:
        querystring['minVitaminB5'] = minvitaminb5
    if maxcarbs:
        querystring['maxCarbs'] = maxcarbs
    if maxvitaminc:
        querystring['maxVitaminC'] = maxvitaminc
    if maxphosphorus:
        querystring['maxPhosphorus'] = maxphosphorus
    if mincalories:
        querystring['minCalories'] = mincalories
    if intolerances:
        querystring['intolerances'] = intolerances
    if minvitaminb2:
        querystring['minVitaminB2'] = minvitaminb2
    if minselenium:
        querystring['minSelenium'] = minselenium
    if minvitamink:
        querystring['minVitaminK'] = minvitamink
    if minmagnesium:
        querystring['minMagnesium'] = minmagnesium
    if maxsodium:
        querystring['maxSodium'] = maxsodium
    if maxvitaminb2:
        querystring['maxVitaminB2'] = maxvitaminb2
    if maxsugar:
        querystring['maxSugar'] = maxsugar
    if maxvitamind:
        querystring['maxVitaminD'] = maxvitamind
    if instructionsrequired:
        querystring['instructionsRequired'] = instructionsrequired
    if maxfolicacid:
        querystring['maxFolicAcid'] = maxfolicacid
    if ranking:
        querystring['ranking'] = ranking
    if maxprotein:
        querystring['maxProtein'] = maxprotein
    if miniodine:
        querystring['minIodine'] = miniodine
    if maxvitaminb3:
        querystring['maxVitaminB3'] = maxvitaminb3
    if minvitaminb6:
        querystring['minVitaminB6'] = minvitaminb6
    if type:
        querystring['type'] = type
    if mincholine:
        querystring['minCholine'] = mincholine
    if maxvitamine:
        querystring['maxVitaminE'] = maxvitamine
    if equipment:
        querystring['equipment'] = equipment
    if minvitaminb1:
        querystring['minVitaminB1'] = minvitaminb1
    if minvitamina:
        querystring['minVitaminA'] = minvitamina
    if maxvitaminb1:
        querystring['maxVitaminB1'] = maxvitaminb1
    if minvitaminb12:
        querystring['minVitaminB12'] = minvitaminb12
    if minpotassium:
        querystring['minPotassium'] = minpotassium
    if maxiron:
        querystring['maxIron'] = maxiron
    if mincopper:
        querystring['minCopper'] = mincopper
    if maxfluoride:
        querystring['maxFluoride'] = maxfluoride
    if minvitamind:
        querystring['minVitaminD'] = minvitamind
    if maxsaturatedfat:
        querystring['maxSaturatedFat'] = maxsaturatedfat
    if maxselenium:
        querystring['maxSelenium'] = maxselenium
    if maxvitaminb12:
        querystring['maxVitaminB12'] = maxvitaminb12
    if mincholesterol:
        querystring['minCholesterol'] = mincholesterol
    if minprotein:
        querystring['minProtein'] = minprotein
    if minfolicacid:
        querystring['minFolicAcid'] = minfolicacid
    if maxzinc:
        querystring['maxZinc'] = maxzinc
    if minfat:
        querystring['minFat'] = minfat
    if maxcholine:
        querystring['maxCholine'] = maxcholine
    if minvitaminc:
        querystring['minVitaminC'] = minvitaminc
    if minfluoride:
        querystring['minFluoride'] = minfluoride
    if maxalcohol:
        querystring['maxAlcohol'] = maxalcohol
    if author:
        querystring['author'] = author
    if minmanganese:
        querystring['minManganese'] = minmanganese
    if minsugar:
        querystring['minSugar'] = minsugar
    if cuisine:
        querystring['cuisine'] = cuisine
    if minalcohol:
        querystring['minAlcohol'] = minalcohol
    if includeingredients:
        querystring['includeIngredients'] = includeingredients
    if minsaturatedfat:
        querystring['minSaturatedFat'] = minsaturatedfat
    if diet:
        querystring['diet'] = diet
    if maxvitaminb6:
        querystring['maxVitaminB6'] = maxvitaminb6
    if minphosphorus:
        querystring['minPhosphorus'] = minphosphorus
    if maxvitamink:
        querystring['maxVitaminK'] = maxvitamink
    if maxfat:
        querystring['maxFat'] = maxfat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_recipes_by_nutrients(limitlicense: bool=None, minprotein: int=0, minvitaminc: int=0, minselenium: int=0, random: bool=None, maxfluoride: int=50, maxvitaminb5: int=50, maxvitaminb3: int=50, maxiodine: int=50, mincarbs: int=0, maxcalories: int=250, minalcohol: int=0, maxcopper: int=50, maxcholine: int=50, maxvitaminb6: int=50, miniron: int=0, maxmanganese: int=50, minsodium: int=0, minsugar: int=0, maxfat: int=20, mincholine: int=0, maxvitaminc: int=50, maxvitaminb2: int=50, minvitaminb12: int=0, maxfolicacid: int=50, minzinc: int=0, offset: int=0, maxprotein: int=100, mincalories: int=0, mincaffeine: int=0, minvitamind: int=0, maxvitamine: int=50, minvitaminb2: int=0, minfiber: int=0, minfolate: int=0, minmanganese: int=0, maxpotassium: int=50, maxsugar: int=50, maxcaffeine: int=50, maxcholesterol: int=50, maxsaturatedfat: int=50, minvitaminb3: int=0, maxfiber: int=50, maxphosphorus: int=50, minpotassium: int=0, maxselenium: int=50, maxcarbs: int=100, mincalcium: int=0, mincholesterol: int=0, minfluoride: int=0, maxvitamind: int=50, maxvitaminb12: int=50, miniodine: int=0, maxzinc: int=50, minsaturatedfat: int=0, minvitaminb1: int=0, maxfolate: int=50, minfolicacid: int=0, maxmagnesium: int=50, minvitamink: int=0, minmagnesium: int=0, maxsodium: int=50, maxalcohol: int=50, maxcalcium: int=50, maxvitamina: int=50, maxvitamink: int=50, minvitaminb5: int=0, maxiron: int=50, mincopper: int=0, maxvitaminb1: int=50, number: int=10, minvitamina: int=0, minphosphorus: int=0, minvitaminb6: int=0, minfat: int=5, minvitamine: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a set of recipes that adhere to the given nutrient limits. All the found recipes will have macro nutrients within the calories, protein, fat, and carbohydrate limits."
    limitlicense: Whether the recipes should have an open license that allows for displaying with proper attribution.
        minprotein: The minimum number of protein in grams the recipe must have.
        minvitaminc: The minimum number of Vitamin C milligrams the recipe must have.
        minselenium: The minimum number of selenium in grams the recipe must have.
        random: If true, every request will give you a random set of recipes within the requested limits.
        maxfluoride: The maximum number of fluoride in milligrams the recipe can have.
        maxvitaminb5: The maximum number of Vitamin B5 in milligrams the recipe can have.
        maxvitaminb3: The maximum number of Vitamin B3 in milligrams the recipe can have.
        maxiodine: The maximum number of iodine in grams the recipe must have.
        mincarbs: The minimum number of carbohydrates in grams the recipe must have.
        maxcalories: The maximum number of calories the recipe can have.
        minalcohol: The minimum number of alcohol in grams the recipe must have.
        maxcopper: The maximum number of copper in milligrams the recipe must have.
        maxcholine: The maximum number of choline in milligrams the recipe can have.
        maxvitaminb6: The maximum number of Vitamin B6 in milligrams the recipe can have.
        miniron: The minimum number of iron in milligrams the recipe must have.
        maxmanganese: The maximum number of manganese in milligrams the recipe can have.
        minsodium: The minimum number of sodium in milligrams the recipe must have.
        minsugar: The minimum number of sugar in grams the recipe must have.
        maxfat: The maximum number of fat in grams the recipe can have.
        mincholine: The minimum number of choline in milligrams the recipe must have.
        maxvitaminc: The maximum number of Vitamin C in milligrams the recipe can have.
        maxvitaminb2: The maximum number of Vitamin B2 in milligrams the recipe must have.
        minvitaminb12: The minimum number of Vitamin B12 in micrograms the recipe must have.
        maxfolicacid: The maximum number of folic acid in grams the recipe must have.
        minzinc: The minimum number of zinc in milligrams the recipe must have.
        offset: The offset number for paging in the interval [0,990].
        maxprotein: The maximum number of protein in grams the recipe can have.
        mincalories: The minimum number of calories the recipe must have.
        mincaffeine: The minimum number of milligrams of caffeine the recipe must have.
        minvitamind: The minimum number of Vitamin D in micrograms the recipe must have.
        maxvitamine: The maximum number of Vitamin E in milligrams the recipe must have.
        minvitaminb2: The minimum number of Vitamin B2 in milligrams the recipe must have.
        minfiber: The minimum number of fiber in grams the recipe must have.
        minfolate: The minimum number of folate in grams the recipe must have.
        minmanganese: The minimum number of manganese in milligrams the recipe must have.
        maxpotassium: The maximum number of potassium in milligrams the recipe can have.
        maxsugar: The maximum number of sugar in grams the recipe must have.
        maxcaffeine: The maximum number of alcohol in grams the recipe must have.
        maxcholesterol: The maximum number of cholesterol in milligrams the recipe must have.
        maxsaturatedfat: The maximum number of saturated fat in grams the recipe must have.
        minvitaminb3: The minimum number of Vitamin B3 in milligrams the recipe must have.
        maxfiber: The maximum number of fiber in grams the recipe must have.
        maxphosphorus: The maximum number of phosphorus in milligrams the recipe can have.
        minpotassium: The minimum number of potassium in milligrams the recipe must have.
        maxselenium: The maximum number of selenium in grams the recipe must have.
        maxcarbs: The maximum number of carbohydrates in grams the recipe can have.
        mincalcium: The minimum number of calcium in milligrams the recipe must have.
        mincholesterol: The minimum number of cholesterol in milligrams the recipe must have.
        minfluoride: The minimum number of fluoride in milligrams the recipe must have.
        maxvitamind: The maximum number of Vitamin D in micrograms the recipe must have.
        maxvitaminb12: The maximum number of Vitamin B12 in micrograms the recipe must have.
        miniodine: The minimum number of Iodine in grams the recipe must have.
        maxzinc: The maximum number of zinc in milligrams the recipe can have.
        minsaturatedfat: The minimum number of saturated fat in grams the recipe must have.
        minvitaminb1: The minimum number of Vitamin B1 in milligrams the recipe must have.
        maxfolate: The maximum number of folate in grams the recipe must have.
        minfolicacid: The minimum number of folic acid in grams the recipe must have.
        maxmagnesium: The maximum number of magnesium in milligrams the recipe can have.
        minvitamink: The minimum number of Vitamin K in micrograms the recipe must have.
        minmagnesium: The minimum number of magnesium in milligrams the recipe must have.
        maxsodium: The maximum number of sodium in milligrams the recipe must have.
        maxalcohol: The maximum number of alcohol in grams the recipe must have.
        maxcalcium: The maximum number of calcium in milligrams the recipe must have.
        maxvitamina: The maximum number of Vitamin A in IU the recipe must have.
        maxvitamink: The maximum number of Vitamin K in micrograms the recipe must have.
        minvitaminb5: The minimum number of Vitamin B5 in milligrams the recipe must have.
        maxiron: The maximum number of iron in milligrams the recipe can have.
        mincopper: The minimum number of copper in milligrams the recipe must have.
        maxvitaminb1: The maximum number of Vitamin B1 in milligrams the recipe must have.
        number: The number of expected results in the interval [1,10].
        minvitamina: The minimum number of Vitamin A in IU the recipe must have.
        minphosphorus: The minimum number of phosphorus in milligrams the recipe must have.
        minvitaminb6: The minimum number of Vitamin B6 in milligrams the recipe must have.
        minfat: The minimum number of fat in grams the recipe must have.
        minvitamine: The minimum number of Vitamin E in milligrams the recipe must have.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/findByNutrients"
    querystring = {}
    if limitlicense:
        querystring['limitLicense'] = limitlicense
    if minprotein:
        querystring['minProtein'] = minprotein
    if minvitaminc:
        querystring['minVitaminC'] = minvitaminc
    if minselenium:
        querystring['minSelenium'] = minselenium
    if random:
        querystring['random'] = random
    if maxfluoride:
        querystring['maxFluoride'] = maxfluoride
    if maxvitaminb5:
        querystring['maxVitaminB5'] = maxvitaminb5
    if maxvitaminb3:
        querystring['maxVitaminB3'] = maxvitaminb3
    if maxiodine:
        querystring['maxIodine'] = maxiodine
    if mincarbs:
        querystring['minCarbs'] = mincarbs
    if maxcalories:
        querystring['maxCalories'] = maxcalories
    if minalcohol:
        querystring['minAlcohol'] = minalcohol
    if maxcopper:
        querystring['maxCopper'] = maxcopper
    if maxcholine:
        querystring['maxCholine'] = maxcholine
    if maxvitaminb6:
        querystring['maxVitaminB6'] = maxvitaminb6
    if miniron:
        querystring['minIron'] = miniron
    if maxmanganese:
        querystring['maxManganese'] = maxmanganese
    if minsodium:
        querystring['minSodium'] = minsodium
    if minsugar:
        querystring['minSugar'] = minsugar
    if maxfat:
        querystring['maxFat'] = maxfat
    if mincholine:
        querystring['minCholine'] = mincholine
    if maxvitaminc:
        querystring['maxVitaminC'] = maxvitaminc
    if maxvitaminb2:
        querystring['maxVitaminB2'] = maxvitaminb2
    if minvitaminb12:
        querystring['minVitaminB12'] = minvitaminb12
    if maxfolicacid:
        querystring['maxFolicAcid'] = maxfolicacid
    if minzinc:
        querystring['minZinc'] = minzinc
    if offset:
        querystring['offset'] = offset
    if maxprotein:
        querystring['maxProtein'] = maxprotein
    if mincalories:
        querystring['minCalories'] = mincalories
    if mincaffeine:
        querystring['minCaffeine'] = mincaffeine
    if minvitamind:
        querystring['minVitaminD'] = minvitamind
    if maxvitamine:
        querystring['maxVitaminE'] = maxvitamine
    if minvitaminb2:
        querystring['minVitaminB2'] = minvitaminb2
    if minfiber:
        querystring['minFiber'] = minfiber
    if minfolate:
        querystring['minFolate'] = minfolate
    if minmanganese:
        querystring['minManganese'] = minmanganese
    if maxpotassium:
        querystring['maxPotassium'] = maxpotassium
    if maxsugar:
        querystring['maxSugar'] = maxsugar
    if maxcaffeine:
        querystring['maxCaffeine'] = maxcaffeine
    if maxcholesterol:
        querystring['maxCholesterol'] = maxcholesterol
    if maxsaturatedfat:
        querystring['maxSaturatedFat'] = maxsaturatedfat
    if minvitaminb3:
        querystring['minVitaminB3'] = minvitaminb3
    if maxfiber:
        querystring['maxFiber'] = maxfiber
    if maxphosphorus:
        querystring['maxPhosphorus'] = maxphosphorus
    if minpotassium:
        querystring['minPotassium'] = minpotassium
    if maxselenium:
        querystring['maxSelenium'] = maxselenium
    if maxcarbs:
        querystring['maxCarbs'] = maxcarbs
    if mincalcium:
        querystring['minCalcium'] = mincalcium
    if mincholesterol:
        querystring['minCholesterol'] = mincholesterol
    if minfluoride:
        querystring['minFluoride'] = minfluoride
    if maxvitamind:
        querystring['maxVitaminD'] = maxvitamind
    if maxvitaminb12:
        querystring['maxVitaminB12'] = maxvitaminb12
    if miniodine:
        querystring['minIodine'] = miniodine
    if maxzinc:
        querystring['maxZinc'] = maxzinc
    if minsaturatedfat:
        querystring['minSaturatedFat'] = minsaturatedfat
    if minvitaminb1:
        querystring['minVitaminB1'] = minvitaminb1
    if maxfolate:
        querystring['maxFolate'] = maxfolate
    if minfolicacid:
        querystring['minFolicAcid'] = minfolicacid
    if maxmagnesium:
        querystring['maxMagnesium'] = maxmagnesium
    if minvitamink:
        querystring['minVitaminK'] = minvitamink
    if minmagnesium:
        querystring['minMagnesium'] = minmagnesium
    if maxsodium:
        querystring['maxSodium'] = maxsodium
    if maxalcohol:
        querystring['maxAlcohol'] = maxalcohol
    if maxcalcium:
        querystring['maxCalcium'] = maxcalcium
    if maxvitamina:
        querystring['maxVitaminA'] = maxvitamina
    if maxvitamink:
        querystring['maxVitaminK'] = maxvitamink
    if minvitaminb5:
        querystring['minVitaminB5'] = minvitaminb5
    if maxiron:
        querystring['maxIron'] = maxiron
    if mincopper:
        querystring['minCopper'] = mincopper
    if maxvitaminb1:
        querystring['maxVitaminB1'] = maxvitaminb1
    if number:
        querystring['number'] = number
    if minvitamina:
        querystring['minVitaminA'] = minvitamina
    if minphosphorus:
        querystring['minPhosphorus'] = minphosphorus
    if minvitaminb6:
        querystring['minVitaminB6'] = minvitaminb6
    if minfat:
        querystring['minFat'] = minfat
    if minvitamine:
        querystring['minVitaminE'] = minvitamine
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def taste_by_id(is_id: int, normalize: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a recipe's taste. The tastes supported are sweet, salty, sour, bitter, savory, and fatty. These tastes are between 0 and 100 while the spiciness value is in scoville on an open scale of 0 and above.
		
		Every ingredient has each of these values and it is weighted by how much they contribute to the recipe. Spiciness is taking the weight of the spicy ingredient and multiplying it with its scoville amount. Of course, taste is also very personal and it depends on how it is prepared so all of the values should only give you an indication of how the dish tastes."
    id: The recipe id.
        normalize: Normalize to the strongest taste.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/tasteWidget.json"
    querystring = {}
    if normalize:
        querystring['normalize'] = normalize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_recipes(query: str, maxvitaminb6: int=100, minfolate: int=0, instructionsrequired: bool=None, maxvitamina: int=100, number: int=10, minzinc: int=0, offset: int=0, maxselenium: int=100, maxcopper: int=100, maxcholine: int=100, diet: str='vegetarian', minselenium: int=0, maxpotassium: int=100, maxfat: int=100, maxphosphorus: int=100, minmagnesium: int=0, maxvitamine: int=100, minvitaminb3: int=0, mincholesterol: int=0, fillingredients: bool=None, maxcarbs: int=100, maxalcohol: int=100, maxcalcium: int=100, minvitamine: int=0, maxprotein: int=100, minsugar: int=0, maxfolicacid: int=100, minvitamink: int=0, miniron: int=0, titlematch: str='Crock Pot', mincalcium: int=0, maxsugar: int=100, addrecipeinformation: bool=None, maxfluoride: int=100, limitlicense: bool=None, minvitamina: int=0, maxsaturatedfat: int=100, minfluoride: int=0, minmanganese: int=0, minvitaminc: int=0, minsaturatedfat: int=0, maxmanganese: int=100, maxvitaminb2: int=100, equipment: str='pan', mincarbs: int=10, minvitaminb1: int=0, maxiron: int=100, maxvitamind: int=100, maxvitaminb3: int=100, mincaffeine: int=0, maxvitamink: int=100, excludeingredients: str='eggs', minvitaminb6: int=0, maxfolate: int=100, intolerances: str='gluten', maxcaffeine: int=100, type: str='main course', includeingredients: str='tomato,cheese', mincholine: int=0, minfat: int=10, minfolicacid: int=0, maxvitaminb12: int=100, author: str=None, maxvitaminc: int=100, minpotassium: int=0, maxmagnesium: int=100, maxfiber: int=100, miniodine: int=0, cuisine: str='italian', ranking: int=2, minalcohol: int=0, mincopper: int=0, excludecuisine: str='greek', maxzinc: int=100, minphosphorus: int=0, maxvitaminb5: int=100, minvitamind: int=0, minvitaminb2: int=0, minsodium: int=0, maxsodium: int=100, minprotein: int=10, minvitaminb12: int=0, mincalories: int=50, maxcholesterol: int=100, maxcalories: int=800, minvitaminb5: int=0, minfiber: int=0, maxiodine: int=100, maxvitaminb1: int=100, tags: str=None, maxreadytime: int=20, recipeboxid: int=None, sort: str='calories', sortdirection: str='asc', ignorepantry: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search through thousands of recipes using advanced filtering and ranking. NOTE: Since this method combines searching by query, by ingredients, and by nutrients into one endpoint, each request counts as 3 requests."
    query: The recipe search query.
        maxvitaminb6: The maximum number of milligrams of Vitamin B6 the recipe can have.
        minfolate: The minimum number of micrograms of folate the recipe must have.
        instructionsrequired: Whether the recipes must have instructions.
        maxvitamina: The maximum number of IU of Vitamin A the recipe can have.
        number: The number of results to return (between 1 and 100).
        minzinc: The minimum number of milligrams of zinc the recipe must have.
        offset: The number of results to skip (between 0 and 900).
        maxselenium: The maximum number of micrograms of selenium the recipe can have.
        maxcopper: The maximum number of milligrams of copper the recipe can have.
        maxcholine: The maximum number of milligrams of choline the recipe can have.
        diet: The diet to which the recipes must be compliant. Possible values are: pescetarian, lacto vegetarian, ovo vegetarian, vegan, paleo, primal, and vegetarian.
        minselenium: The minimum number of micrograms of selenium the recipe must have.
        maxpotassium: The maxnimum number of milligrams of potassium the recipe can have.
        maxfat: The maximum number of grams of fat the recipe can have.
        maxphosphorus: The maximum number of milligrams of phosphorus the recipe can have.
        minmagnesium: The minimum number of milligrams of magnesium the recipe must have.
        maxvitamine: The maximum number of milligrams of Vitamin E the recipe can have.
        minvitaminb3: The minimum number of milligrams of Vitamin B3 the recipe must have.
        mincholesterol: The minimum number of milligrams of cholesterol the recipe must have.
        fillingredients: Add information about the used and missing ingredients in each recipe.
        maxcarbs: The maximum number of grams of carbohydrates the recipe can have.
        maxalcohol: The maximum number of grams of alcohol the recipe can have.
        maxcalcium: The maximum number of milligrams of calcium the recipe can have.
        minvitamine: The minimum number of milligrams of Vitamin E the recipe must have.
        maxprotein: The maximum number of grams of protein the recipe can have.
        minsugar: The minimum number of grams of sugar the recipe must have.
        maxfolicacid: The maximum number of micrograms of folic acid the recipe can have.
        minvitamink: The minimum number of micrograms of Vitamin K the recipe must have.
        miniron: The minimum number of milligrams of iron the recipe must have.
        titlematch: A string that the recipes must contain in their titles.
        mincalcium: The minimum number of milligrams of calcium the recipe must have.
        maxsugar: The maximum number of grams of sugar the recipe can have.
        addrecipeinformation: If set to true, you get more information about the recipes returned. This saves the calls to get recipe information.
        maxfluoride: The maximum number of milligrams of fluoride the recipe can have.
        limitlicense: Whether the recipes should have an open license that allows for displaying with proper attribution.
        minvitamina: The minimum number of IU of Vitamin A the recipe must have.
        maxsaturatedfat: The maximum number of grams of saturated fat the recipe can have.
        minfluoride: The minimum number of milligrams of fluoride the recipe must have.
        minmanganese: The minimum number of milligrams of manganese the recipe must have.
        minvitaminc: The minimum number of milligrams of Vitamin C the recipe must have.
        minsaturatedfat: The minimum number of grams of saturated fat the recipe must have.
        maxmanganese: The maximum number of milligrams of manganese the recipe can have.
        maxvitaminb2: The maximum number of milligrams of Vitamin B2 the recipe can have.
        equipment: The equipment required. Multiple values will be interpreted as 'or'. For example, value could be \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"blender, frying pan, bowl\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
        mincarbs: The minimum number of grams of carbohydrates the recipe must have.
        minvitaminb1: The minimum number of milligrams of Vitamin B1 the recipe must have.
        maxiron: The maximum number of milligrams of iron the recipe can have.
        maxvitamind: The maximum number of micrograms of Vitamin D the recipe can have.
        maxvitaminb3: The maximum number of milligrams of Vitamin B3 the recipe can have.
        mincaffeine: The minimum number of milligrams of caffeine the recipe must have.
        maxvitamink: The maximum number of micrograms of Vitamin K the recipe can have.
        excludeingredients: An comma-separated list of ingredients that must not be contained in the recipes.
        minvitaminb6: The minimum number of milligrams of Vitamin B6 the recipe must have.
        maxfolate: The maximum number of micrograms of folate the recipe can have.
        intolerances: A comma-separated list of intolerances. All found recipes must not have ingredients that could cause problems for people with one of the given tolerances. Possible values are: dairy, egg, gluten, peanut, sesame, seafood, shellfish, soy, sulfite, tree nut, and wheat.
        maxcaffeine: The maximum number of milligrams of caffeine the recipe can have.
        type: The type of the recipes. One of the following: main course, side dish, dessert, appetizer, salad, bread, breakfast, soup, beverage, sauce, or drink.
        includeingredients: A comma-separated list of ingredients that should/must be contained in the recipe.
        mincholine: The minimum number of milligrams of choline the recipe must have.
        minfat: The minimum number of grams of fat the recipe must have.
        minfolicacid: The minimum number of micrograms of folic acid the recipe must have.
        maxvitaminb12: The maximum number of micrograms of Vitamin B12 the recipe can have.
        author: The username of the recipe author.
        maxvitaminc: The maximum number of milligrams of Vitamin C the recipe can have.
        minpotassium: The minimum number of milligrams of potassium the recipe must have.
        maxmagnesium: The maximum number of milligrams of magnesium the recipe can have.
        maxfiber: The maximum number of grams of fiber the recipe can have.
        miniodine: The minimum number of micrograms of iodine the recipe must have.
        cuisine: The cuisine(s) of the recipes. One or more (comma separated) of the following: african, chinese, japanese, korean, vietnamese, thai, indian, british, irish, french, italian, mexican, spanish, middle eastern, jewish, american, cajun, southern, greek, german, nordic, eastern european, caribbean, or latin american.
        ranking: Whether to minimize missing ingredients (0), maximize used ingredients (1) first, or rank recipes by relevance (2).
        minalcohol: The minimum number of grams of alcohol the recipe must have.
        mincopper: The minimum number of milligrams of copper the recipe must have.
        excludecuisine: The cuisine(s) the recipes must not match. One or more, comma separated (will be interpreted as 'AND'). See a full [list of supported cuisines](https://spoonacular.com/food-api/docs#Cuisines).
        maxzinc: The maximum number of milligrams of zinc the recipe can have.
        minphosphorus: The minimum number of milligrams of phosphorus the recipe must have.
        maxvitaminb5: The maximum number of milligrams of Vitamin B5 the recipe can have.
        minvitamind: The minimum number of micrograms of Vitamin D the recipe must have.
        minvitaminb2: The minimum number of milligrams of Vitamin B2 the recipe must have.
        minsodium: The minimum number of milligrams of sodium the recipe must have.
        maxsodium: The maximum number of milligrams of sodium the recipe can have.
        minprotein: The minimum number of grams of protein the recipe must have.
        minvitaminb12: The minimum number of micrograms of Vitamin B12 the recipe must have.
        mincalories: The minimum number of calories the recipe must have.
        maxcholesterol: The maximum number of milligrams of cholesterol the recipe can have.
        maxcalories: The maximum number of calories the recipe can have.
        minvitaminb5: The minimum number of milligrams of Vitamin B5 the recipe must have.
        minfiber: The minimum number of grams of fiber the recipe must have.
        maxiodine: The maximum number of micrograms of iodine the recipe can have.
        maxvitaminb1: The maximum number of milligrams of Vitamin B1 the recipe can have.
        tags: User defined tags that have to match. The author param has to be set.
        maxreadytime: The maximum time in minutes it should take to prepare and cook the recipe.
        recipeboxid: The id of the recipe box to which the search should be limited to.
        sort: The strategy to sort recipes by. See the [full list of supported sorting options.](https://spoonacular.com/food-api/docs#Recipe-Sorting-Options)
        sortdirection: The direction in which to sort. Must be either 'asc' (ascending) or 'desc' (descending).
        ignorepantry: Whether to ignore typical pantry items, such as water, salt, flour, etc.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/complexSearch"
    querystring = {'query': query, }
    if maxvitaminb6:
        querystring['maxVitaminB6'] = maxvitaminb6
    if minfolate:
        querystring['minFolate'] = minfolate
    if instructionsrequired:
        querystring['instructionsRequired'] = instructionsrequired
    if maxvitamina:
        querystring['maxVitaminA'] = maxvitamina
    if number:
        querystring['number'] = number
    if minzinc:
        querystring['minZinc'] = minzinc
    if offset:
        querystring['offset'] = offset
    if maxselenium:
        querystring['maxSelenium'] = maxselenium
    if maxcopper:
        querystring['maxCopper'] = maxcopper
    if maxcholine:
        querystring['maxCholine'] = maxcholine
    if diet:
        querystring['diet'] = diet
    if minselenium:
        querystring['minSelenium'] = minselenium
    if maxpotassium:
        querystring['maxPotassium'] = maxpotassium
    if maxfat:
        querystring['maxFat'] = maxfat
    if maxphosphorus:
        querystring['maxPhosphorus'] = maxphosphorus
    if minmagnesium:
        querystring['minMagnesium'] = minmagnesium
    if maxvitamine:
        querystring['maxVitaminE'] = maxvitamine
    if minvitaminb3:
        querystring['minVitaminB3'] = minvitaminb3
    if mincholesterol:
        querystring['minCholesterol'] = mincholesterol
    if fillingredients:
        querystring['fillIngredients'] = fillingredients
    if maxcarbs:
        querystring['maxCarbs'] = maxcarbs
    if maxalcohol:
        querystring['maxAlcohol'] = maxalcohol
    if maxcalcium:
        querystring['maxCalcium'] = maxcalcium
    if minvitamine:
        querystring['minVitaminE'] = minvitamine
    if maxprotein:
        querystring['maxProtein'] = maxprotein
    if minsugar:
        querystring['minSugar'] = minsugar
    if maxfolicacid:
        querystring['maxFolicAcid'] = maxfolicacid
    if minvitamink:
        querystring['minVitaminK'] = minvitamink
    if miniron:
        querystring['minIron'] = miniron
    if titlematch:
        querystring['titleMatch'] = titlematch
    if mincalcium:
        querystring['minCalcium'] = mincalcium
    if maxsugar:
        querystring['maxSugar'] = maxsugar
    if addrecipeinformation:
        querystring['addRecipeInformation'] = addrecipeinformation
    if maxfluoride:
        querystring['maxFluoride'] = maxfluoride
    if limitlicense:
        querystring['limitLicense'] = limitlicense
    if minvitamina:
        querystring['minVitaminA'] = minvitamina
    if maxsaturatedfat:
        querystring['maxSaturatedFat'] = maxsaturatedfat
    if minfluoride:
        querystring['minFluoride'] = minfluoride
    if minmanganese:
        querystring['minManganese'] = minmanganese
    if minvitaminc:
        querystring['minVitaminC'] = minvitaminc
    if minsaturatedfat:
        querystring['minSaturatedFat'] = minsaturatedfat
    if maxmanganese:
        querystring['maxManganese'] = maxmanganese
    if maxvitaminb2:
        querystring['maxVitaminB2'] = maxvitaminb2
    if equipment:
        querystring['equipment'] = equipment
    if mincarbs:
        querystring['minCarbs'] = mincarbs
    if minvitaminb1:
        querystring['minVitaminB1'] = minvitaminb1
    if maxiron:
        querystring['maxIron'] = maxiron
    if maxvitamind:
        querystring['maxVitaminD'] = maxvitamind
    if maxvitaminb3:
        querystring['maxVitaminB3'] = maxvitaminb3
    if mincaffeine:
        querystring['minCaffeine'] = mincaffeine
    if maxvitamink:
        querystring['maxVitaminK'] = maxvitamink
    if excludeingredients:
        querystring['excludeIngredients'] = excludeingredients
    if minvitaminb6:
        querystring['minVitaminB6'] = minvitaminb6
    if maxfolate:
        querystring['maxFolate'] = maxfolate
    if intolerances:
        querystring['intolerances'] = intolerances
    if maxcaffeine:
        querystring['maxCaffeine'] = maxcaffeine
    if type:
        querystring['type'] = type
    if includeingredients:
        querystring['includeIngredients'] = includeingredients
    if mincholine:
        querystring['minCholine'] = mincholine
    if minfat:
        querystring['minFat'] = minfat
    if minfolicacid:
        querystring['minFolicAcid'] = minfolicacid
    if maxvitaminb12:
        querystring['maxVitaminB12'] = maxvitaminb12
    if author:
        querystring['author'] = author
    if maxvitaminc:
        querystring['maxVitaminC'] = maxvitaminc
    if minpotassium:
        querystring['minPotassium'] = minpotassium
    if maxmagnesium:
        querystring['maxMagnesium'] = maxmagnesium
    if maxfiber:
        querystring['maxFiber'] = maxfiber
    if miniodine:
        querystring['minIodine'] = miniodine
    if cuisine:
        querystring['cuisine'] = cuisine
    if ranking:
        querystring['ranking'] = ranking
    if minalcohol:
        querystring['minAlcohol'] = minalcohol
    if mincopper:
        querystring['minCopper'] = mincopper
    if excludecuisine:
        querystring['excludeCuisine'] = excludecuisine
    if maxzinc:
        querystring['maxZinc'] = maxzinc
    if minphosphorus:
        querystring['minPhosphorus'] = minphosphorus
    if maxvitaminb5:
        querystring['maxVitaminB5'] = maxvitaminb5
    if minvitamind:
        querystring['minVitaminD'] = minvitamind
    if minvitaminb2:
        querystring['minVitaminB2'] = minvitaminb2
    if minsodium:
        querystring['minSodium'] = minsodium
    if maxsodium:
        querystring['maxSodium'] = maxsodium
    if minprotein:
        querystring['minProtein'] = minprotein
    if minvitaminb12:
        querystring['minVitaminB12'] = minvitaminb12
    if mincalories:
        querystring['minCalories'] = mincalories
    if maxcholesterol:
        querystring['maxCholesterol'] = maxcholesterol
    if maxcalories:
        querystring['maxCalories'] = maxcalories
    if minvitaminb5:
        querystring['minVitaminB5'] = minvitaminb5
    if minfiber:
        querystring['minFiber'] = minfiber
    if maxiodine:
        querystring['maxIodine'] = maxiodine
    if maxvitaminb1:
        querystring['maxVitaminB1'] = maxvitaminb1
    if tags:
        querystring['tags'] = tags
    if maxreadytime:
        querystring['maxReadyTime'] = maxreadytime
    if recipeboxid:
        querystring['recipeBoxId'] = recipeboxid
    if sort:
        querystring['sort'] = sort
    if sortdirection:
        querystring['sortDirection'] = sortdirection
    if ignorepantry:
        querystring['ignorePantry'] = ignorepantry
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dish_pairing_for_wine(wine: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a dish that goes well with a given wine."
    wine: The type of wine that should be paired, e.g. \"merlot\", \"riesling\", or \"malbec\".
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/wine/dishes"
    querystring = {'wine': wine, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_classification_url(imageurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Classify a food image."
    
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/images/classify"
    querystring = {'imageUrl': imageurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quick_answer(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Answer a nutrition related natural language question."
    q: The nutrition-related question.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/quickAnswer"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_meal_plan(timeframe: str='day', targetcalories: int=2000, diet: str='vegetarian', exclude: str='shellfish, olives', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a meal plan with three meals per day (breakfast, lunch, and dinner)."
    timeframe: Either for one 'day' or an entire 'week'.
        targetcalories: What is the caloric target for one day? The meal plan generator will try to get as close as possible to that goal.
        diet: Enter a diet that the meal plan has to adhere to, e.g. "vegetarian", "vegan", "paleo" etc.
        exclude: A comma-separated list of allergens or ingredients that must be excluded.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/mealplans/generate"
    querystring = {}
    if timeframe:
        querystring['timeFrame'] = timeframe
    if targetcalories:
        querystring['targetCalories'] = targetcalories
    if diet:
        querystring['diet'] = diet
    if exclude:
        querystring['exclude'] = exclude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_food_trivia(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns random food trivia."
    
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/trivia/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete_menu_item_search(query: str, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate suggestions for menu items based on a (partial) query. The matches will be found by looking in the title only."
    query: The (partial) search query.
        number: The number of results to return. Must be between 1 and 25.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/menuItems/suggest"
    querystring = {'query': query, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_menu_items(query: str, offset: int=0, number: int=10, mincalories: int=0, maxcalories: int=5000, minprotein: int=0, maxprotein: int=100, minfat: int=0, maxfat: int=100, mincarbs: int=0, maxcarbs: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search menu items (such as McDonalds Big Mac)"
    query: The search query.
        offset: The search offset.
        number: The number of results to return.
        mincalories: The minimum number of calories the menu item can have
        maxcalories: The maximum number of calories the menu item can have
        minprotein: The minimum number of grams of protein the menu item can have
        maxprotein: The maximum number of grams of protein the menu item can have
        minfat: The minimum number of grams of fat the menu item can have.
        maxfat: The maximum number of grams of fat the menu item can have.
        mincarbs: The minimum number of grams of carbs the menu item can have.
        maxcarbs: The maximum number of grams of carbs the menu item can have.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/menuItems/search"
    querystring = {'query': query, }
    if offset:
        querystring['offset'] = offset
    if number:
        querystring['number'] = number
    if mincalories:
        querystring['minCalories'] = mincalories
    if maxcalories:
        querystring['maxCalories'] = maxcalories
    if minprotein:
        querystring['minProtein'] = minprotein
    if maxprotein:
        querystring['maxProtein'] = maxprotein
    if minfat:
        querystring['minFat'] = minfat
    if maxfat:
        querystring['maxFat'] = maxfat
    if mincarbs:
        querystring['minCarbs'] = mincarbs
    if maxcarbs:
        querystring['maxCarbs'] = maxcarbs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_grocery_products(query: str, maxcalories: int=5000, minprotein: int=0, maxprotein: int=100, minfat: int=0, maxfat: int=100, mincarbs: int=0, maxcarbs: int=100, mincalories: int=0, offset: int=0, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search packaged food products like frozen pizza and snickers bars."
    query: The search query.
        maxcalories: The maximum number of calories the product can have.
        minprotein: The minimum number of grams of protein the product can have.
        maxprotein: The maximum number of grams of protein the product can have.
        minfat: The minimum number of grams of fat the product can have.
        maxfat: The maximum number of grams of fat the product can have.
        mincarbs: The minimum number of grams of carbs the product can have.
        maxcarbs: The maximum number of grams of carbs the product can have.
        mincalories: The minimum number of calories the product can have.
        offset: The number of results to skip, defaults to 0.
        number: The number of results to retrieve, defaults to 10.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/products/search"
    querystring = {'query': query, }
    if maxcalories:
        querystring['maxCalories'] = maxcalories
    if minprotein:
        querystring['minProtein'] = minprotein
    if maxprotein:
        querystring['maxProtein'] = maxprotein
    if minfat:
        querystring['minFat'] = minfat
    if maxfat:
        querystring['maxFat'] = maxfat
    if mincarbs:
        querystring['minCarbs'] = mincarbs
    if maxcarbs:
        querystring['maxCarbs'] = maxcarbs
    if mincalories:
        querystring['minCalories'] = mincalories
    if offset:
        querystring['offset'] = offset
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summarize_recipe(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Summarize the recipe in a short text."
    id: The id of the recipe that should be summarized.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/summary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_analyzed_recipe_instructions(is_id: str, stepbreakdown: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an analyzed breakdown of a recipe's instructions. Each step is enriched with the ingredients and the equipment that is used."
    id: The id of the recipe.
        stepbreakdown: Whether to break down the recipe steps even more.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/analyzedInstructions"
    querystring = {}
    if stepbreakdown:
        querystring['stepBreakdown'] = stepbreakdown
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ingredient_substitutes(ingredientname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ingredient substitutes by ingredient name."
    ingredientname: The name of the ingredient you want to replace.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/ingredients/substitutes"
    querystring = {'ingredientName': ingredientname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ingredient_substitutes_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for substitutes for a given ingredient."
    id: The id of the ingredient you want substitutes for.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/ingredients/{is_id}/substitutes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ingredient_information(is_id: int, unit: str='grams', amount: int=150, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use an ingredient id to get all available information about an ingredient, such as its image and supermarket aisle."
    id: The id of the food (ingredient).
        unit: The unit for the given amount.
        amount: The amount of that food.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/ingredients/{is_id}/information"
    querystring = {}
    if unit:
        querystring['unit'] = unit
    if amount:
        querystring['amount'] = amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_food_videos(query: str='chicken soup', type: str=None, minlength: int=0, maxlength: int=999, number: int=10, cuisine: str=None, includeingredients: str='chicken', excludeingredients: str='mustard', offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find recipe and other food related videos."
    query: The search query.
        type: The type of the recipe, e.g. one of the following: main course, side dish, dessert, appetizer, salad, bread, breakfast, soup, beverage, sauce, or drink.
        minlength: Minimum video length in seconds.
        maxlength: Maximum video length in seconds.
        number: The number of results [1,100].
        cuisine: The cuisine of the videos, e.g. one of the following: african, chinese, japanese, korean, vietnamese, thai, indian, british, irish, french, italian, mexican, spanish, middle eastern, jewish, american, cajun, southern, greek, german, nordic, eastern european, caribbean, latin american
        includeingredients: Ingredients the recipe videos must contain.
        excludeingredients: Ingredients the recipe videos must not contain.
        offset: The offset in the result set [0,900].
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/videos/search"
    querystring = {}
    if query:
        querystring['query'] = query
    if type:
        querystring['type'] = type
    if minlength:
        querystring['minLength'] = minlength
    if maxlength:
        querystring['maxLength'] = maxlength
    if number:
        querystring['number'] = number
    if cuisine:
        querystring['cuisine'] = cuisine
    if includeingredients:
        querystring['includeingredients'] = includeingredients
    if excludeingredients:
        querystring['excludeingredients'] = excludeingredients
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def guess_nutrition_by_dish_name(title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Guess the macro nutrients of a dish given its title."
    title: The title of the dish.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/guessNutrition"
    querystring = {'title': title, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete_product_search(query: str, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate suggestions for grocery products based on a (partial) query. The matches will be found by looking in the title only."
    query: The (partial) search query.
        number: The number of results to return. Must be between 1 and 25.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/food/products/suggest"
    querystring = {'query': query, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price_breakdown_by_id(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a recipe's price breakdown data."
    id: The recipe id.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{is_id}/priceBreakdownWidget.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_recipes_by_ingredients(ingredients: str, number: int=5, limitlicense: bool=None, ignorepantry: bool=None, ranking: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find recipes that use as many of the given ingredients as possible and have as little as possible missing ingredients. This is a whats in your fridge API endpoint."
    ingredients: A comma-separated list of ingredients that the recipes should contain.
        number: The maximal number of recipes to return (default = 5).
        limitlicense: Whether to only show recipes with an attribution license.
        ignorepantry: Whether to ignore pantry ingredients such as water, salt, flour etc..
        ranking: Whether to maximize used ingredients (1) or minimize missing ingredients (2) first.
        
    """
    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/findByIngredients"
    querystring = {'ingredients': ingredients, }
    if number:
        querystring['number'] = number
    if limitlicense:
        querystring['limitLicense'] = limitlicense
    if ignorepantry:
        querystring['ignorePantry'] = ignorepantry
    if ranking:
        querystring['ranking'] = ranking
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

