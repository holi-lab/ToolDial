import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shopper(description: str='"Made out of used wine barrels from our Frostwatch label. These barrels no longer impart the flavors needed for our wines, but that doesn’t mean they can’t be enjoyed. We glue three staves together and add four foot pads for added stability. This piece is oil finished and great for home display or unique serving method.', name: str='Oak Barrell Stave Trays', category: str='Barrell', dimensions: str='37 feet by 8.5 feet', price: int=45, is_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "items listed"
    
    """
    url = f"https://ravproducts.p.rapidapi.com/shopall"
    querystring = {}
    if description:
        querystring['description'] = description
    if name:
        querystring['name'] = name
    if category:
        querystring['category'] = category
    if dimensions:
        querystring['dimensions'] = dimensions
    if price:
        querystring['price'] = price
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ravproducts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

