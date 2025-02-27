import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def naver_shop_category_search(catid: str, sellerfilter: str='total', pagenum: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search with Naver shopping category number."
    
    """
    url = f"https://naver-shop-data-scraper.p.rapidapi.com/naver/category"
    querystring = {'catId': catid, }
    if sellerfilter:
        querystring['sellerFilter'] = sellerfilter
    if pagenum:
        querystring['pageNum'] = pagenum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "naver-shop-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def naver_shop_search_request(query: str, pagenum: str='1', sellerfilter: str='total', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Naver shop search request"
    
    """
    url = f"https://naver-shop-data-scraper.p.rapidapi.com/naver/search"
    querystring = {'query': query, }
    if pagenum:
        querystring['pageNum'] = pagenum
    if sellerfilter:
        querystring['sellerFilter'] = sellerfilter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "naver-shop-data-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

