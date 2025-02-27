import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product(page: str=None, title: str=None, asin: str=None, barcode: int=9780439625593, search: str=None, category: str=None, mpn: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to retrieve product data."
    page: Page number for any search query (10 results per page)
        title: Product Name
        asin: ASIN
        barcode: UPC, EAN or ISBN number
        search: Any search term or phrase
        mpn: Manufacturer Part Number
        
    """
    url = f"https://barcode-lookup.p.rapidapi.com/v3/products"
    querystring = {}
    if page:
        querystring['page'] = page
    if title:
        querystring['title'] = title
    if asin:
        querystring['asin'] = asin
    if barcode:
        querystring['barcode'] = barcode
    if search:
        querystring['search'] = search
    if category:
        querystring['category'] = category
    if mpn:
        querystring['mpn'] = mpn
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

