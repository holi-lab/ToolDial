import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getgenders(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of genders"
    
    """
    url = f"https://v1-streetwear.p.rapidapi.com/genders"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v1-streetwear.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproducts(limit: str, page: str=None, brand: str=None, name: str=None, product: str=None, colorway: str=None, releasedate: str=None, styleid: str=None, sort: str=None, gender: str=None, releaseyear: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of products"
    limit: Number of products to return (min = 10 & max = 100)
        page: Pagination
        brand: Product brand name
        name: Title of product
        product: product type
        colorway: Colorway of product
        releasedate: Release date of product
        styleid: Style ID of product
        sort: Sort results by field
        gender: Gender of product
        releaseyear: Release year of product
        
    """
    url = f"https://v1-streetwear.p.rapidapi.com/products"
    querystring = {'limit': limit, }
    if page:
        querystring['page'] = page
    if brand:
        querystring['brand'] = brand
    if name:
        querystring['name'] = name
    if product:
        querystring['product'] = product
    if colorway:
        querystring['colorway'] = colorway
    if releasedate:
        querystring['releaseDate'] = releasedate
    if styleid:
        querystring['styleId'] = styleid
    if sort:
        querystring['sort'] = sort
    if gender:
        querystring['gender'] = gender
    if releaseyear:
        querystring['releaseYear'] = releaseyear
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v1-streetwear.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getproductbyid(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single product"
    productid: ID of product to return
        
    """
    url = f"https://v1-streetwear.p.rapidapi.com/products/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v1-streetwear.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbrands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of brands"
    
    """
    url = f"https://v1-streetwear.p.rapidapi.com/brands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v1-streetwear.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

