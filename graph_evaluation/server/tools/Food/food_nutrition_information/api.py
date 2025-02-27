import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_food_by_id(fooid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a single food item by an FDC ID."
    
    """
    url = f"https://food-nutrition-information.p.rapidapi.com/food/{fooid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-nutrition-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_foods_using_keywords(query: str, brandowner: str='Kar Nut Products Company', pagesize: str='1', pagenumber: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for foods using keywords."
    
    """
    url = f"https://food-nutrition-information.p.rapidapi.com/foods/search"
    querystring = {'query': query, }
    if brandowner:
        querystring['brandOwner'] = brandowner
    if pagesize:
        querystring['pageSize'] = pagesize
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "food-nutrition-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

