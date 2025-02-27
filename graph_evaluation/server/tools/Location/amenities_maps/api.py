import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_amenity_locations(amenity: str, mylat: str, mylon: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    amenity: toilet, atm, bank, postoffice, bakery, bar, bench, bicyclerental, cafe, convenience, drinkingwater, fastfood, fuel, parking, pharmacy, playground, pub, subwayentrance, supermarket
        
    """
    url = f"https://community-amenities-maps.p.rapidapi.com/amenimapi.php"
    querystring = {'amenity': amenity, 'mylat': mylat, 'mylon': mylon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-amenities-maps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

