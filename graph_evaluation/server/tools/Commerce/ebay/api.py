import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_details(product_id: int, country: str='germany', country_code: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the product details for a given product id and a specific country.
		Default country is `United States`.
		Specify country with country name or country code.
		
		Allowed countries:
		Default: `us`
		- Germany (de)
		- France (fr)
		- Australia (au)
		- Austria (at)
		- Canada (ca)
		- Hong Kong (hk)
		- Ireland (ie)
		- Italy (it)
		- Malaysia (my)
		- Netherlands (nl)
		- Singapore (sg)
		- Switzerland (ch)
		- United Kingdom (uk)"
    product_id: ID of the product. Can be obtained from the url of the product or by using the `/search` endpoint.
        country: Valid country to return offers for.
Valid values are in description of this endpoint.
Default: `united states`.
        country_code: Country code of the valid country to return offers for.
Valid values are in description of this endpoint.
Default: `us`.
        
    """
    url = f"https://ebay32.p.rapidapi.com/product/{product_id}"
    querystring = {}
    if country:
        querystring['country'] = country
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_products(search_query: str, country: str='germany', page: int=1, country_code: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for products on Ebay in specific country.
		Returns the `last_page` that can be scraped for the specific `search_query`.
		Default country is `United States`.
		Specify country with country name or country code.
		
		Allowed countries:
		Default: `us`
		- Germany (de)
		- France (fr)
		- Australia (au)
		- Austria (at)
		- Canada (ca)
		- Hong Kong (hk)
		- Ireland (ie)
		- Italy (it)
		- Malaysia (my)
		- Netherlands (nl)
		- Singapore (sg)
		- Switzerland (ch)
		- United Kingdom (uk)"
    search_query: Search Query used in Ebay search
        country: Valid country to return offers for.
Valid values are in description of this endpoint.
Default: `united states`.
        page: Results page to return.
Default: `1`.
        country_code: Country code of the valid country to return offers for.
Valid values are in description of this endpoint.
Default: `us`.
        
    """
    url = f"https://ebay32.p.rapidapi.com/search/{search_query}"
    querystring = {}
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ebay32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

