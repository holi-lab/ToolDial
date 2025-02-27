import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbrands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of brands"
    
    """
    url = f"https://the-sneaker-database.p.rapidapi.com/brands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-sneaker-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgenders(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of genders"
    
    """
    url = f"https://the-sneaker-database.p.rapidapi.com/genders"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-sneaker-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(limit: str, page: str=None, query: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of matching sneakers"
    limit: Number of sneakers to return (min = 10 & max = 100)
        page: Pagination (defaults to 0 if not provided)
        query: Text to search for
        
    """
    url = f"https://the-sneaker-database.p.rapidapi.com/search"
    querystring = {'limit': limit, }
    if page:
        querystring['page'] = page
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-sneaker-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsneakers(limit: str, gender: str=None, silhouette: str=None, colorway: str=None, releaseyear: str=None, page: str=None, releasedate: str=None, sku: str=None, sort: str=None, name: str=None, brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of sneakers"
    limit: Number of sneakers to return (min = 10 & max = 100)
        gender: Gender of sneakers
        silhouette: Silhouette
        colorway: Colorway of sneakers
        releaseyear: Release year of sneakers
        page: Pagination (defaults to 0 if not provided)
        releasedate: Release date of sneakers
        sku: SKU of sneakers
        sort: Sort results by field
        name: Name of sneakers
        brand: Sneaker brand name
        
    """
    url = f"https://the-sneaker-database.p.rapidapi.com/sneakers"
    querystring = {'limit': limit, }
    if gender:
        querystring['gender'] = gender
    if silhouette:
        querystring['silhouette'] = silhouette
    if colorway:
        querystring['colorway'] = colorway
    if releaseyear:
        querystring['releaseYear'] = releaseyear
    if page:
        querystring['page'] = page
    if releasedate:
        querystring['releaseDate'] = releasedate
    if sku:
        querystring['sku'] = sku
    if sort:
        querystring['sort'] = sort
    if name:
        querystring['name'] = name
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-sneaker-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsneakerbyid(sneakerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single sneaker"
    sneakerid: Id of sneaker to return
        
    """
    url = f"https://the-sneaker-database.p.rapidapi.com/sneakers/{sneakerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-sneaker-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

