import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gameslist(region: str, start: int, count: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint will List all the Games which are on Special Discount Offers."
    
    """
    url = f"https://steamgames-special-offers.p.rapidapi.com/games_list/"
    querystring = {'region': region, 'start': start, 'count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steamgames-special-offers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gamesdata(app_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will give you the Games Data if you provide the APP_ID.
		You can use the GamesList endpoint to get the list of APP_IDs and Feed it to this Endpoint to get the Games Data. Sample Details are given below:
		`{
		    "discount": "-40%",
		    "original_price": "$49.99",
		    "price": "$29.99",
		    "title": "God of War",
		    "url": "https://store.steampowered.com/app/1593500/"
		}`"
    
    """
    url = f"https://steamgames-special-offers.p.rapidapi.com/games_data/"
    querystring = {'app_id': app_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steamgames-special-offers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

