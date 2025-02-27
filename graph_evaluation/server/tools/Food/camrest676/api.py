import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bookrestaurant(pricerange: str, area: str, food: str='spanish', restaurantname: str='McDonalds', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the restaurants the user is looking for"
    pricerange: pricerange desired by the user
        area: area where the restaurant is located
        food: type of food
        restaurantname: a restaurant
        
    """
    url = f"https://camrest676.p.rapidapi.com/bookrestaurant"
    querystring = {'pricerange': pricerange, 'area': area, }
    if food:
        querystring['food'] = food
    if restaurantname:
        querystring['restaurantName'] = restaurantname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "camrest676.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

