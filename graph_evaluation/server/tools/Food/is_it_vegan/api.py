import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def isitvegan(product_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns json-formatted response with information about the product.
		
		*Request:*
		GET /{product-name}
		
		*Returns:*
		`{`
		`  "palmOil": "YES" | "NO" | "UNKNOWN" | "MAYBE", `
		`  "status": "OK" | "NOT-FOUND" | "ERROR", `
		`  "vegan": "YES" | "NO" | "UNKNOWN" | "MAYBE" ,`
		`  "vegetarian": "YES" | "NO" | "UNKNOWN" | "MAYBE"`
		`}`"
    
    """
    url = f"https://is-it-vegan.p.rapidapi.com/{product_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-it-vegan.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

