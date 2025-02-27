import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def item_detail_api(is_id: str, region: str='US', locale: str='en_US', currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All details of a single item. We fetch this item data from desktop item page in real time. We collect and format response you needed.
		
		If any information we missing, please let me know. Thanks."
    region: The \\\\\\\"Region\\\\\\\" parameter is used to obtain precise shipping costs and other promotions or deals for a specific geographical location. Only values from an approved list are accepted after being validated. To view the full list of supported geographical options, please refer to the \\\\\\\"Country List\\\\\\\" found in the \\\\\\\"Basic information\\\\\\\" group endpoint.
        locale: The \\\\\\\"Locale\\\\\\\" parameter is used to display titles and other content in the selected language. Only values from an approved list are accepted after being validated. To view the full list of supported language options, please refer to the \\\\\\\"Locale List\\\\\\\" found in the \\\\\\\"Basic information\\\\\\\" group endpoint.
        currency: The \\\\\\\"Currency\\\\\\\" parameter is utilized to display the price of a product in a specific currency. Only values from an approved list are accepted after being validated. To view the full list of supported currencies, please refer to the \\\\\\\"Currency List\\\\\\\" found in the \\\\\\\"Basic information\\\\\\\" group endpoint.
        
    """
    url = f"https://aliexpress-api.p.rapidapi.com/aliexpress/v1/simply/item/{is_id}"
    querystring = {}
    if region:
        querystring['region'] = region
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_api(keyword: str, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search items by keyword. Please put urlencoded keyword in this api"
    keyword: urlencoded string
        
    """
    url = f"https://aliexpress-api.p.rapidapi.com/aliexpress/v1/search/{keyword}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shop_item_list(seller_id: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get shop items by seller id"
    
    """
    url = f"https://aliexpress-api.p.rapidapi.com/aliexpress/v1/shop/{seller_id}/products/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shop_information(seller_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get shop information by seller id"
    
    """
    url = f"https://aliexpress-api.p.rapidapi.com/aliexpress/v1/shop/{seller_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Currency" parameter is used to display the price of a product in the selected currency. Only valid values from the supported list are accepted. To see the complete list of supported currencies, use the designated endpoint."
    
    """
    url = f"https://aliexpress-api.p.rapidapi.com/aliexpress/v1/currenry"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Region" parameter is used to specify the location of the recipient. This is used to calculate accurate shipping costs and provide location-specific promotions or deals. Only valid values from the supported list are accepted. To see the complete list of supported countries or regions, use the designated endpoint."
    
    """
    url = f"https://aliexpress-api.p.rapidapi.com/aliexpress/v1/country"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locale_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "Locale" parameter is used to display titles and other content in the selected language. Only values from an approved list are accepted after being validated. To view the full list of supported language options, please refer to the "Locale List" found in the "Basic information" group endpoint."
    
    """
    url = f"https://aliexpress-api.p.rapidapi.com/aliexpress/v1/locale"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

