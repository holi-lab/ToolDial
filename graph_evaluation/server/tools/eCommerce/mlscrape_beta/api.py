import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_beta(url: str, proxy: str='enabled', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**This is an older version of the API - a newer version of this API is available!**
		
		### Overview
		
		This API returns structured data extracted from a product page, which is specified by a URL. The URL must be valid and begin with "http"/"https", otherwise an error will be returned (see the section below on errors for more information). The API returns the following set of data:
		
		**Name**: As a string, or null if unavailable.
		E.g "Triban RC 500 Disk Road Bike".
		
		**Price**: As a string, or null if unavailable.
		E.g "£549.99".
		
		**Main Image URL**: As a string, or null if unavailable
		
		**Specification information**: 
		As an object, with key-value pairs that correspond to the values in the detected specification table.
		
		 For example, if the specification table contains a row with "Brand" in the first column and "Triban" in the second column, then the specification object will contain, ```"brand": "Triban"```.
		
		If no specification information is detected on the product page, then the specification object will be an empty object.
		
		**Identifiers**:
		The identifiers object contains any of the following identifiers if they are detected on the product page: SKU, UPC/EAN, MPN, ISBN. If none of them are detected, then the identifiers object will be empty.
		E.g  
		``` 
		       {
		           "upc": "0622356231244", 
		           "sku": "184321572622"
		       }
		```
		
		### Rotating proxy
		To enable rotating proxies for your request, add `proxy=enabled` to your API request as a query parameter.
		Any requests with proxies enabled will use up **2 requests** from your API quota.
		
		### Errors
		If an error occurs while processing your request, one of the following error codes will be returned in an error object:
		
		**1**
		**Invalid URL** - Please ensure that the URL provided is a valid URL, and that it begins with http/https. 
		
		**2**
		**Timeout error** - The product page took too long to load.
		
		**3**
		**Failed to load the product page** - The API failed to load the product page. This could be due to a number of reasons, for example, our servers may have been blocked by the owner of the product page."
    
    """
    url = f"https://mlscrape-beta.p.rapidapi.com/product"
    querystring = {'url': url, }
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlscrape-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_v1(url: str, proxy: str='enabled', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Overview
		
		This API returns structured data extracted from a product page, which is specified by a URL. The product page should be the webpage of a specific product on an online shop (rather than, for example, a search results page).  The URL must be valid and begin with "http"/"https", otherwise an error will be returned (see the section below on errors for more information). The API returns the following set of data:
		
		**Name**: As a string, or null if unavailable.
		E.g "Triban RC 500 Disk Road Bike".
		
		**Price**: Broken down into 3 parts: The (ISO 4217) currency code, the textual, and the numeric representation of the price. If the price was not detected, null is returned instead.
		E.g
		``` 
		       {
		           "currency": "GBP",
		           "text": "£549.99", 
		           "value": 549.99
		       }
		```
		
		**Main Image URL**: As a string, or null if unavailable
		
		**Description**:
		The description of the product as a string, or null if unavailable. 
		
		**Availability**:
		The availability status of the product, which is one of the following (or null if undetected):
		- "In Stock"
		- "Discontinued"
		- "Limited" (i.e the store's stock levels for the product are low)
		- "Out Of Stock"
		
		**Specification information**: 
		As an object, with key-value pairs that correspond to the values in the detected specification table.
		
		 For example, if the specification table contains a row with "Brand" in the first column and "Triban" in the second column, then the specification object will contain, ```"brand": "Triban"```.
		
		If no specification information is detected on the product page, then the specification object will be an empty object.
		
		**Identifiers**:
		The identifiers object contains any of the following identifiers if they are detected on the product page: SKU, UPC/EAN, MPN, ISBN. If none of them are detected, then the identifiers object will be empty.
		E.g  
		``` 
		       {
		           "upc": "0622356231244", 
		           "sku": "184321572622"
		       }
		```
		
		### Rotating proxy
		To enable rotating proxies for your request, add `proxy=enabled` to your API request as a query parameter.
		Any requests with proxies enabled will use up **2 requests** from your API quota.
		
		### Errors
		If an error occurs while processing your request, one of the following error codes will be returned in an error object:
		
		**1**
		**Invalid URL** - Please ensure that the URL provided is a valid URL, and that it begins with http/https. 
		
		**2**
		**Timeout error** - The product page took too long to load.
		
		**3**
		**Failed to load the product page** - The API failed to load the product page. This could be due to a number of reasons, for example, our servers may have been blocked by the owner of the product page."
    
    """
    url = f"https://mlscrape-beta.p.rapidapi.com/v1/product"
    querystring = {'url': url, }
    if proxy:
        querystring['proxy'] = proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mlscrape-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

