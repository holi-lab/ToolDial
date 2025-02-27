import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def keyword_search(keyword: str, page: int, maincat: str='damen', sortby: str='sale', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve keyword search results."
    maincat: Category used for the search, valid values are \"katalog\" (default), \"damen\", \"herren\" and \"kinder\".
        sortby: Default is \"most-popular\" (if left blank), other valid values are \"activation-date\",  \"price-asc\", \"price-desc\" and \"sale\".
        
    """
    url = f"https://axesso-zalando-data-service.p.rapidapi.com/zal/zalando-search-by-keyword?"
    querystring = {'keyword': keyword, 'page': page, }
    if maincat:
        querystring['mainCat'] = maincat
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-zalando-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query product details information."
    
    """
    url = f"https://axesso-zalando-data-service.p.rapidapi.com/zal/zalando-lookup-product"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-zalando-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

