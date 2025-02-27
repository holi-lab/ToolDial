import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def similaritems_v2(materialid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the products customers who bought this product also bought"
    materialid: Here are the some of sample Product Ids B01FGZ8VMC, B0199T0DN6, B0199SZVV6, B01DZ5OBGA, B0199U54G6, B0184ZD432
        
    """
    url = f"https://inoor-shaik-bms-api-similar-items-v1.p.rapidapi.com/v2/{materialid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "inoor-shaik-bms-api-similar-items-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

