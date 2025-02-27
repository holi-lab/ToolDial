import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_bsr(categoryid: str, page: str='1', url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the best seller directly from Amazon. 
		You can read this article that explains how to use the endpoint. => https://www.halios-data.com/post/how-to-get-the-best-seller-ranking-in-a-few-seconds"
    categoryid: Represents the category-id get from the category endpoint where the attribute isBSRDepartment is true
        url: URL get from amazon website by example 

> https://www.amazon.com/Best-Sellers-Appliances/zgbs/appliances/ref=zg_bs_nav_0
        
    """
    url = f"https://amazon24.p.rapidapi.com/api/bsr/{categoryid}"
    querystring = {}
    if page:
        querystring['page'] = page
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_todaydeals(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the products display in the today's deals from Amazon US only."
    url: Here indicated the deal_details_url, retrieved in the deal_docs collection.


        
    """
    url = f"https://amazon24.p.rapidapi.com/api/todaydeals"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_product_productid(productid: str, country: str='US', producturl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents a product, each product is identifier by ID/asin"
    productid: The Amazon product ID
        country: Represents the language wished to display the product, by default the system sets US
        producturl: add the product URL that you want to scrap. Warning, the productId in the path of the URL must be equal to 0
        
    """
    url = f"https://amazon24.p.rapidapi.com/api/product/{productid}"
    querystring = {}
    if country:
        querystring['country'] = country
    if producturl:
        querystring['productURL'] = producturl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_data_management_error_product(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When the scraping of one product failed, the system saved it and tried to scrape it the second time, thanks to a cron task."
    
    """
    url = f"https://amazon24.p.rapidapi.com/api/data-management/error/product"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_product_productid_competitors(productid: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents the competitors for one product"
    productid: The Amazon product ID
        country: Represents the language wished to display the reviews, by default the system sets US.
        
    """
    url = f"https://amazon24.p.rapidapi.com/api/product/{productid}/competitors"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_product(refinementid: str=None, country: str='US', filter: str=None, categoryid: str='aps', page: int=1, keyword: str='iphone', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search a product by key words, categoryId or filters"
    country: Represents the language wished to display the products and filters.
        filter: Represents the refinments proposes by Amazon to affine or filter the products, don't forget to init it,  when you change the category, the key words or the country.
        categoryid: Represents the categoryID to filter the products, you can add the category ID retrieves from category end point or the id from featuredCategories  returns by this end point .
        page: Represents the page wished.
        keyword: Represents the key words to found the products wished.
        
    """
    url = f"https://amazon24.p.rapidapi.com/api/product"
    querystring = {}
    if refinementid:
        querystring['refinementID'] = refinementid
    if country:
        querystring['country'] = country
    if filter:
        querystring['filter'] = filter
    if categoryid:
        querystring['categoryID'] = categoryid
    if page:
        querystring['page'] = page
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_review_productid(productid: str, sortby: str='helpful', country: str='US', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents the reviews for one product"
    productid: The Amazon product ID
        country: Represents the language wished to display the reviews, by default the system sets US.
        page: Represents the page wished. By default the system sets 1.
        
    """
    url = f"https://amazon24.p.rapidapi.com/api/review/{productid}"
    querystring = {}
    if sortby:
        querystring['sortBy'] = sortby
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_data_management_error_product_productid(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When the scraping of one product failed, the system saved it and tried to scrape it the second time, thanks to a cron task."
    productid: The Amazon product ID
        
    """
    url = f"https://amazon24.p.rapidapi.com/api/data-management/error/product/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_category(country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This resource represents a category in the system. You can choose a specific country, by default, if you don t select a country, the system sets US  "
    country: Represents the language wished to display the product, by default the system sets US
        
    """
    url = f"https://amazon24.p.rapidapi.com/api/category"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon24.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

