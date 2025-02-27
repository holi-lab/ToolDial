import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crops_list(subcategory: str='De invierno', family: str='Poaceae', commonname: str='Trigo', speciename: str='Triticum', consistency: str='herbácea', cropcycle: str='anual', fruittype: str='cariopsis', category: str='Cereales de grano', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns back all crops that are grown in Spain, being possible to filter them through query string (see Optional Parameters)."
    
    """
    url = f"https://crops.p.rapidapi.com/"
    querystring = {}
    if subcategory:
        querystring['subcategory'] = subcategory
    if family:
        querystring['family'] = family
    if commonname:
        querystring['commonName'] = commonname
    if speciename:
        querystring['specieName'] = speciename
    if consistency:
        querystring['consistency'] = consistency
    if cropcycle:
        querystring['cropCycle'] = cropcycle
    if fruittype:
        querystring['fruitType'] = fruittype
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crops.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

