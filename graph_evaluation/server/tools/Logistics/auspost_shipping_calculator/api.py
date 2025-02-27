import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def postcode_search(search: str, limit: str='10', state: str='VIC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a search string of a full or partial location name or postcode to return:
		
		```
		{
		  "localities": {
		    "locality": {
		      "category": "Delivery Area",
		      "id": 5358,
		      "latitude": -37.81443733,
		      "location": "EAST MELBOURNE",
		      "longitude": 144.9825846,
		      "postcode": 3002,
		      "state": "VIC"
		    }
		  }
		}
		```"
    limit: Optional parameter to increase result set from the default 10 to up to 100 results
        state: Optional parameter to  filter search results to a specific state, valid inputs are:
- NSW
- NT
- QLD
- SA
- TAS
- VIC
- WA
- 
        
    """
    url = f"https://auspost-shipping-calculator.p.rapidapi.com/postage/auspost/postcode/search"
    querystring = {'search': search, }
    if limit:
        querystring['limit'] = limit
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "auspost-shipping-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

