import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geo_dictionary(q: str, api_token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use in 'search' endpoint to filter influencers by their GEO or GEO of their audience"
    q: Search query, name of the city or country to filter influencers
        api_token: To buy API Token register on http://deep.social/ and check our prices here http://deep.social/prices
        
    """
    url = f"https://deepsocial.p.rapidapi.com/v1/accounts/geo"
    querystring = {'q': q, 'api_token': api_token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "deepsocial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

