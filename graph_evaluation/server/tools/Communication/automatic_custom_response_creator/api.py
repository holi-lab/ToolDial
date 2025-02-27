import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_response(business_type: str, business_brand: str, avis: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "There are 3 params necessary : 
		
		- First - Add a review "avis" (ex. "Merci pour votre accueil, c'était parfait")
		- Then - Add a "business_type" (ex. "Restaurant")
		- Finaly - Add a "business_brand" (ex. "Fuzi")"
    
    """
    url = f"https://automatic-custom-response-creator.p.rapidapi.com/get_response"
    querystring = {'business_type': business_type, 'business_brand': business_brand, 'avis': avis, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "automatic-custom-response-creator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

