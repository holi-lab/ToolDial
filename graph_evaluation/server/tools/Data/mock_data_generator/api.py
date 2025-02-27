import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mock_address_details(count: int=15, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**GET:**
		**/address?count={value}**
		
		- You ​can mock country, city, state, zipCode, latitude, longitude by using this endpoint.
		- Value must be from 0 to 50  only..
		- By using this end point you can only generate mock data upto 50 per request only"
    count: this value should be between 1 to 50
        
    """
    url = f"https://mock-data-generator.p.rapidapi.com/address"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mock-data-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mock_products(count: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**GET:**
		**/products?count={value}**
		
		- You ​can mock product, price, color by using this endpoint.
		- Value must be from 0 to 50  only..
		- By using this end point you can only generate mock data upto 50 per request only"
    count: Number should be between 1 to 50
        
    """
    url = f"https://mock-data-generator.p.rapidapi.com/products"
    querystring = {'count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mock-data-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mock_person_details(count: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**GET:**
		**/persons?count={value}**
		
		- You ​can mock person  name and mail by using this endpoint.
		- Value must be from 0 to 50  only..
		- By using this end point you can only generate mock data upto 50 per request only"
    count: this value should be between 1 to 50
        
    """
    url = f"https://mock-data-generator.p.rapidapi.com/persons/"
    querystring = {'count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mock-data-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

