import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nivito_au(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Looking for a sleek and stylish kitchen faucet to complete your modern kitchen design? Look no further than the Nivito Kitchen Faucet. This elegant faucet is made from high-quality materials and features a contemporary design that will perfectly complement your kitchen décor. The Nivito Kitchen Faucet is also extremely easy to use, with a single lever control that makes it simple to adjust the water flow and temperature. Plus, the integrated sprayhead provides even coverage for all your cooking and cleaning needs. So if you're looking for a top-of-the-line kitchen faucet that will make your kitchen look even more amazing, choose the Nivito Kitchen Faucet. You won't be disappointed [https://www.nivito.com.au/](https://www.nivito.com.au/)"
    
    """
    url = f"https://nivito-au.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nivito-au.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

