import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def products_detail(country: str, lang: str, productcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of product"
    country: The country code that is gotten from /regions/list endpoint
        lang: The language code that is gotten from /regions/list endpoint
        productcode: The value of articles/code json object returned in /products/list endpoint
        
    """
    url = f"https://apidojo-hm-hennes-mauritz-v1.p.rapidapi.com/products/detail"
    querystring = {'country': country, 'lang': lang, 'productcode': productcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-hm-hennes-mauritz-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list(country: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all categories from H&M"
    country: The country code that is gotten from /regions/list endpoint
        lang: The language code that is gotten from /regions/list endpoint
        
    """
    url = f"https://apidojo-hm-hennes-mauritz-v1.p.rapidapi.com/categories/list"
    querystring = {'country': country, 'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-hm-hennes-mauritz-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list(lang: str, currentpage: int, country: str, pagesize: int, qualities: str=None, fits: str=None, categories: str='men_all', sortby: str=None, collection: str=None, sizes: str=None, colorwithnames: str=None, contexts: str=None, functions: str=None, concepts: str='H&M MAN', descriptivelengths: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products from H&M with options to sort, filter, or search for products by name"
    lang: The language code that is gotten from /regions/list endpoint
        currentpage: The page index to get data, start from 0
        country: The country code that is gotten from /regions/list endpoint
        pagesize: The number of records to return in each page
        qualities: Look for the value in \"facets\" object with \"code\": \"qualities\", pass this param multiple times to filter by multiple qualities
        fits: Look for the value in \"facets\" object with \"code\": \"fits\", pass this param multiple times to filter by multiple fits
        categories: It is tagCodes field gotten from /categories/list endpoint, pass this param multiple times to filter by multiple categories
        sortby: One of the following ascPrice|descPrice|stock|newProduct, default is stock
        collection: Look for the value in \"facets\" object with \"code\": \"collection\", pass this param multiple times to filter by multiple collection
        sizes: Look for the value in \"facets\" object with \"code\": \"sizes\", pass this param multiple times to filter by multiple sizes
        colorwithnames: Look for the value in \"facets\" object with \"code\": \"colorWithNames\", pass this param multiple times to filter by multiple colors
        contexts: Look for the value in \"facets\" object with \"code\": \"contexts\", pass this param multiple times to filter by multiple contexts
        functions: Look for the value in \"facets\" object with \"code\": \"functions\", pass this param multiple times to filter by multiple functions
        concepts: Look for the value in \"facets\" object with \"code\": \"concepts\", pass this param multiple times to filter by multiple concepts
        descriptivelengths: Look for the value in \"facets\" object with \"code\": \"descriptiveLengths\", pass this param multiple times to filter by multiple lengths
        
    """
    url = f"https://apidojo-hm-hennes-mauritz-v1.p.rapidapi.com/products/list"
    querystring = {'lang': lang, 'currentpage': currentpage, 'country': country, 'pagesize': pagesize, }
    if qualities:
        querystring['qualities'] = qualities
    if fits:
        querystring['fits'] = fits
    if categories:
        querystring['categories'] = categories
    if sortby:
        querystring['sortBy'] = sortby
    if collection:
        querystring['collection'] = collection
    if sizes:
        querystring['sizes'] = sizes
    if colorwithnames:
        querystring['colorWithNames'] = colorwithnames
    if contexts:
        querystring['contexts'] = contexts
    if functions:
        querystring['functions'] = functions
    if concepts:
        querystring['concepts'] = concepts
    if descriptivelengths:
        querystring['descriptiveLengths'] = descriptivelengths
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-hm-hennes-mauritz-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_search_by_barcode(gtincodes: str, country: str='us', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search product by barcode"
    gtincodes: The scanned code
        country: The country code
        lang: The language code
        
    """
    url = f"https://apidojo-hm-hennes-mauritz-v1.p.rapidapi.com/products/search-by-barcode"
    querystring = {'gtinCodes': gtincodes, }
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-hm-hennes-mauritz-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

