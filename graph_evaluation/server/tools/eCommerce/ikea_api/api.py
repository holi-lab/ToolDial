import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stores(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of all stores in a specified country."
    countrycode: Can be obtained through the  **Countries** endpoint,
        
    """
    url = f"https://ikea-api.p.rapidapi.com/stores"
    querystring = {'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ikea-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(countrycode: str, languagecode: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of all categories available in a specified country."
    countrycode: Can be obtained through the  **Countries** endpoint,
        languagecode: Can be obtained through the  **Countries** endpoint,
        
    """
    url = f"https://ikea-api.p.rapidapi.com/categories"
    querystring = {'countryCode': countrycode, }
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ikea-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product(countrycode: str, productid: str, languagecode: str='en', store: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up-to-date Information for a specific product."
    countrycode: Can be obtained through the  **Countries** endpoint,
        productid: Can be obtained through the  **Search By Keyword or Category** endpoints,
        languagecode: Can be obtained through the  **Countries** endpoint,
        store: Can be obtained through the  **Stores** endpoint,
        
    """
    url = f"https://ikea-api.p.rapidapi.com/product"
    querystring = {'countryCode': countrycode, 'productID': productid, }
    if languagecode:
        querystring['languageCode'] = languagecode
    if store:
        querystring['store'] = store
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ikea-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_keyword_filters(countrycode: str, keyword: str, languagecode: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Complimentary to the "Search By Keyword" endpoint. Obtain a list of filters available based on a keyword."
    countrycode: Can be obtained through the  **Countries** endpoint,
        languagecode: Can be obtained through the  **Countries** endpoint,
        
    """
    url = f"https://ikea-api.p.rapidapi.com/keywordFilter"
    querystring = {'countryCode': countrycode, 'keyword': keyword, }
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ikea-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_keyword(countrycode: str, keyword: str, filters: str=None, languagecode: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of Ikea products information based on a keyword."
    countrycode: Can be obtained through the  **Countries** endpoint,
        filters: **format**: *parameterId=valueId,parameterId=valueId*
Can be obtained through the  **Search By Keyword Filters** endpoint,
        languagecode: Can be obtained through the  **Countries** endpoint,
        
    """
    url = f"https://ikea-api.p.rapidapi.com/keywordSearch"
    querystring = {'countryCode': countrycode, 'keyword': keyword, }
    if filters:
        querystring['filters'] = filters
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ikea-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_category_filters(categoryid: str, countrycode: str, languagecode: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Complimentary to the "Search By Category" endpoint. Obtain a list of filters available to a specified category."
    categoryid: Can be obtained through the **Categories** endpoint,
        countrycode: Can be obtained through the  **Countries** endpoint,
        languagecode: Can be obtained through the  **Countries** endpoint,
        
    """
    url = f"https://ikea-api.p.rapidapi.com/categoryFilter"
    querystring = {'categoryID': categoryid, 'countryCode': countrycode, }
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ikea-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_category(countrycode: str, categoryid: str, filters: str=None, languagecode: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of Ikea products information based on a specified categoryID."
    countrycode: Can be obtained through the **Countries** endpoint,
        categoryid: Can be obtained through the **Categories** endpoint,
        filters: **Format**: *parameter=value_id,parameter=value_id*
Can be obtained through the **Search By Category Filters** endpoint,
        languagecode: Can be obtained through the **Countries** endpoint,
        
    """
    url = f"https://ikea-api.p.rapidapi.com/categorySearch"
    querystring = {'countryCode': countrycode, 'categoryID': categoryid, }
    if filters:
        querystring['filters'] = filters
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ikea-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain a list of all the countries and languages this API supports."
    
    """
    url = f"https://ikea-api.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ikea-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

