import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbrands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of brands"
    
    """
    url = f"https://v1-sneakers.p.rapidapi.com/v1/brands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v1-sneakers.p.rapidapi.com"
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
    url = f"https://v1-sneakers.p.rapidapi.com/v1/genders"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v1-sneakers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsneakers(limit: str, gender: str=None, page: str=None, brand: str=None, styleid: str=None, colorway: str=None, shoe: str=None, releaseyear: str=None, sort: str=None, releasedate: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of sneakers"
    limit: Number of sneakers to return (min = 10 & max = 100)
        gender: Gender of sneakers
        page: Pagination
        brand: Sneaker brand name
        styleid: Style ID of sneakers
        colorway: Colorway of sneakers
        shoe: Silhouette
        releaseyear: Release year of sneakers
        sort: Sort results by field
        releasedate: Release date of sneakers
        name: Title of sneakers
        
    """
    url = f"https://v1-sneakers.p.rapidapi.com/v1/sneakers"
    querystring = {'limit': limit, }
    if gender:
        querystring['gender'] = gender
    if page:
        querystring['page'] = page
    if brand:
        querystring['brand'] = brand
    if styleid:
        querystring['styleId'] = styleid
    if colorway:
        querystring['colorway'] = colorway
    if shoe:
        querystring['shoe'] = shoe
    if releaseyear:
        querystring['releaseYear'] = releaseyear
    if sort:
        querystring['sort'] = sort
    if releasedate:
        querystring['releaseDate'] = releasedate
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v1-sneakers.p.rapidapi.com"
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
    sneakerid: ID of sneaker to return
        
    """
    url = f"https://v1-sneakers.p.rapidapi.com/v1/sneakers/{sneakerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v1-sneakers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

