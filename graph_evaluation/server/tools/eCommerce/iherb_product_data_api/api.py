import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_solicitation(protocol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a download link containing all over 30,000 products, in json format in a .zip file. Remember, the protocol expires in 24 hours! Download your .zip ASAP!"
    
    """
    url = f"https://iherb-product-data-api.p.rapidapi.com/api/IHerb/solicitations/{protocol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iherb-product-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_by_brand_name(brandname: str, page: int, maxprice: int=999, minrating: int=0, hasstock: bool=None, mindiscountpercent: int=None, isspecial: bool=None, minshippingweight: int=0, istrial: bool=None, hasnewproductflag: bool=None, hasdiscount: bool=None, maxshippingweight: int=999, minprice: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all products by brand, and allows you to filter in all ways!"
    
    """
    url = f"https://iherb-product-data-api.p.rapidapi.com/api/IHerb/brands/{brandname}/products"
    querystring = {'page': page, }
    if maxprice:
        querystring['maxPrice'] = maxprice
    if minrating:
        querystring['minRating'] = minrating
    if hasstock:
        querystring['hasStock'] = hasstock
    if mindiscountpercent:
        querystring['minDiscountPercent'] = mindiscountpercent
    if isspecial:
        querystring['isSpecial'] = isspecial
    if minshippingweight:
        querystring['minShippingWeight'] = minshippingweight
    if istrial:
        querystring['isTrial'] = istrial
    if hasnewproductflag:
        querystring['hasNewProductFlag'] = hasnewproductflag
    if hasdiscount:
        querystring['hasDiscount'] = hasdiscount
    if maxshippingweight:
        querystring['maxShippingWeight'] = maxshippingweight
    if minprice:
        querystring['minPrice'] = minprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iherb-product-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_brands(page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all available brands"
    
    """
    url = f"https://iherb-product-data-api.p.rapidapi.com/api/IHerb/brands"
    querystring = {'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "iherb-product-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

