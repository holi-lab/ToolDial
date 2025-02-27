import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def products_search_by_barcode(upc: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product by barcode"
    upc: The scanned code (UPC)
        
    """
    url = f"https://kohls.p.rapidapi.com/products/search-by-barcode"
    querystring = {'upc': upc, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kohls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stores_list(longitude: int, latitude: int, radius: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List stores near a provided GEO location"
    longitude: The longitude of GEO location
        latitude: The latitude of GEO location
        radius: The radius to look for stores around the GEO location
        
    """
    url = f"https://kohls.p.rapidapi.com/stores/list"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kohls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def qnas_list(productid: str, sort: str='SubmissionTime:desc', offset: int=0, limit: int=6, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List questions and answers relating to a product"
    productid: The value of webID returned in .../products/list endpoint
        sort: One of the followings : LastApprovedAnswerSubmissionTime:desc|LastApprovedAnswerSubmissionTime:asc|SubmissionTime:desc|SubmissionTime:asc
        offset: For paging purpose
        limit: For paging purpose
        
    """
    url = f"https://kohls.p.rapidapi.com/qnas/list"
    querystring = {'ProductId': productid, }
    if sort:
        querystring['Sort'] = sort
    if offset:
        querystring['Offset'] = offset
    if limit:
        querystring['Limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kohls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available categories"
    
    """
    url = f"https://kohls.p.rapidapi.com/categories/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kohls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_detail(webid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of specific product"
    webid: The value of webID returned in .../products/list endpoint
        
    """
    url = f"https://kohls.p.rapidapi.com/products/detail"
    querystring = {'webID': webid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kohls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(productid: str, limit: int=6, sort: str='SubmissionTime:desc', offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews relating to a product"
    productid: The value of webID returned in .../products/list endpoint
        limit: For paging purpose
        sort: One of the followings : SubmissionTime:asc|SubmissionTime:desc|Rating:asc|Rating:desc|Helpfulness:asc|Helpfulness:desc|HasPhotos:asc|HasPhotos:desc|HasVideos:asc|HasVideos:desc
        offset: For paging purpose
        
    """
    url = f"https://kohls.p.rapidapi.com/reviews/list"
    querystring = {'ProductId': productid, }
    if limit:
        querystring['Limit'] = limit
    if sort:
        querystring['Sort'] = sort
    if offset:
        querystring['Offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kohls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list(dimensionvalueid: str='AgeAppropriate:Teens', limit: int=24, keyword: str=None, offset: int=1, sortid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products with options and filters"
    dimensionvalueid: The value of payload/dimensions/dimensionValues/currentDimensionId JSON object returned right in this endpoint. Pass this parameter several times to filter by multiple options. Ex : ...&dimensionValueID=AgeAppropriate:Teens&dimensionValueID=Size:TWIN...
        limit: For paging purpose
        keyword: Any term or phrase to look for relating products. Ex : bikini top
        offset: For paging purpose, starting from 1
        sortid: The value of payload/sorts/ID JSON object returned right in this endpoint.
        
    """
    url = f"https://kohls.p.rapidapi.com/products/list"
    querystring = {}
    if dimensionvalueid:
        querystring['dimensionValueID'] = dimensionvalueid
    if limit:
        querystring['limit'] = limit
    if keyword:
        querystring['keyword'] = keyword
    if offset:
        querystring['offset'] = offset
    if sortid:
        querystring['sortID'] = sortid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kohls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete_deprecating(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions by term or phrase"
    query: Any term or phrase that you are familiar with
        
    """
    url = f"https://kohls.p.rapidapi.com/auto-complete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kohls.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

