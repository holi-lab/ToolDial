import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def receive_the_cocktail_data(cocktail_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive the requested cocktail data set, consisting of:
		
		- cocktail name
		- ingredients
		- how to make
		- how to serve
		
		**The Api gives you a selection of 5 matches from our cocktail database consisting of over 600 different cocktails.**"
    
    """
    url = f"https://cocktail8.p.rapidapi.com/request/"
    querystring = {'cocktail_name': cocktail_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cocktail8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

