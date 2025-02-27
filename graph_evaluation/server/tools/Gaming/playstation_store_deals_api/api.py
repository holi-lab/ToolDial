import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def playstationdeals(count: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "There is only 1 parameter for this API endpoint.
		
		1. playstation_deals/?count=0
		
		count = 0 (Min is 0, starting of the list. Max value depends on the total number of games available.)
		Note: Since its a List of Items, If the maximum number of games available on deals is 771 then you have to enter (771-1) = 770 to get the last game on the deal.
		
		This will provide you with the game data as given below which contains name, price, platform, discount percent, discounted price, total no. of games, etc..:
		
		{
		  "name": "God of War III Remastered",
		  "titleId": "CUSA01623_00",
		  "platform": [
		    "PS4"
		  ],
		  "basePrice": "$19.99",
		  "discountPercent": "-50%",
		  "discountPrice": "$9.99",
		  "url": "https://store.playstation.com/en-us/product/UP9000-CUSA01623_00-0000GODOFWAR3PS4",
		  "Total No. of Games": 771
		}"
    
    """
    url = f"https://playstation-store-deals-api.p.rapidapi.com/playstation_deals/"
    querystring = {'count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "playstation-store-deals-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

