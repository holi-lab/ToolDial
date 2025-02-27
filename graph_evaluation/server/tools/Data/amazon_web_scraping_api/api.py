import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_websites(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the available websites available by country to extract amazon product data."
    
    """
    url = f"https://amazon-web-scraping-api.p.rapidapi.com/websites"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-web-scraping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_products_all_countries(criteria: str, countrycode: str='US', page: int=1, languagecode: str='EN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to search products in Amazon using real time data scraping API."
    criteria: Search term that will be used to search for products on Amazon.
        countrycode: This query parameter represents the country code of the Amazon store where the search will be performed. This parameter should be replaced with the two-letter code of the country that you want to search on Amazon. Default 'US' for United States. Check available list in Get Webistes endpoints.
        page: Page number of results that you want to retrieve. If this parameter is not specified, the results of the first page will be returned.
        
    """
    url = f"https://amazon-web-scraping-api.p.rapidapi.com/products/search"
    querystring = {'criteria': criteria, }
    if countrycode:
        querystring['countryCode'] = countrycode
    if page:
        querystring['page'] = page
    if languagecode:
        querystring['languageCode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-web-scraping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_deals(countrycode: str='US', languagecode: str='EN', categoryid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get amazon deals allowing to filter by category.  The category list is included in the response."
    categoryid: Allows to filter by category ID. By default all the categories are returned. 
        
    """
    url = f"https://amazon-web-scraping-api.p.rapidapi.com/products/deals"
    querystring = {}
    if countrycode:
        querystring['countryCode'] = countrycode
    if languagecode:
        querystring['languageCode'] = languagecode
    if categoryid:
        querystring['categoryId'] = categoryid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-web-scraping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of currently supported languages in the API."
    
    """
    url = f"https://amazon-web-scraping-api.p.rapidapi.com/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-web-scraping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_details_all_countries(asin: str, languagecode: str='EN', countrycode: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the detail of a specific product from Amazon using the ASIN code."
    
    """
    url = f"https://amazon-web-scraping-api.p.rapidapi.com/products/{asin}"
    querystring = {}
    if languagecode:
        querystring['languageCode'] = languagecode
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-web-scraping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_reviews_all_countries(asin: str, languagecode: str='EN', countrycode: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the reviews for specific product using the ASIN code."
    
    """
    url = f"https://amazon-web-scraping-api.p.rapidapi.com/products/{asin}/reviews"
    querystring = {}
    if languagecode:
        querystring['languageCode'] = languagecode
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-web-scraping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

