import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getquotesbycategory(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For valid response try these categories listed below:-
		- Quotes Categories are:-
		
		    age   science  success  time  travel
		
		    wisdom   alone  art  attitude  courage
		
		    culture  dreams  friendship  happiness hope
		
		    humour  imagination  inspirational   life   motivational
		
		    nature  philosophy  poetry   popular  psychology"
    
    """
    url = f"https://quotes-villa.p.rapidapi.com/quotes/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotes-villa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

