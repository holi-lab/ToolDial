import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_search_zipcode_keyword(keyword: str, zipcode: str, limit: int=60, offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "provide list of products based on user's zipcode
		response also includes available filters/sort and categories"
    
    """
    url = f"https://weee-grocery-api-sayweee-com-browsing-searching-details.p.rapidapi.com/search"
    querystring = {'keyword': keyword, 'zipcode': zipcode, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weee-grocery-api-sayweee-com-browsing-searching-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details_zipcode_product_id(zipcode: str, product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "provide detailed product information based on the product id and user's zipcode
		product_id should come from the product search endpoint
		response also contains detailed price and availability information"
    
    """
    url = f"https://weee-grocery-api-sayweee-com-browsing-searching-details.p.rapidapi.com/details"
    querystring = {'zipcode': zipcode, 'product_id': product_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "weee-grocery-api-sayweee-com-browsing-searching-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

