import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_nutrition_data(ingr: str, nutrition_type: str='cooking', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns the nutritional analysis for the specified food text by extracting information from a short unstructured food text (usually an ingredient line and returns the following structured data for the text: quantity, measure and food,) and if available: diet, health and allergen labels for the text.  With the built in food logging feature, this allows for change of context. For example, “rice” will normally be matched to raw rice while, with the food logging feature on, it will match to ready to eat ‘cooked rice.’ 
		 
		 <b>Access Point:</b> https://api.edamam.com/api/nutrition-data"
    ingr: The ingredient.
        nutrition_type: Select between the cooking and food logging processor.
        
    """
    url = f"https://edamam-edamam-nutrition-analysis.p.rapidapi.com/api/nutrition-data"
    querystring = {'ingr': ingr, }
    if nutrition_type:
        querystring['nutrition-type'] = nutrition_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edamam-edamam-nutrition-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

