import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_on_amazon(searchquery: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get the particular page including all the products related to your search"
    
    """
    url = f"https://ar7-amazon-scraper-api.p.rapidapi.com/search/{searchquery}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ar7-amazon-scraper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_offers(productid: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get the offers related to a product using the same unique id after /dp/"
    
    """
    url = f"https://ar7-amazon-scraper-api.p.rapidapi.com/products/{productid}/offers"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ar7-amazon-scraper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_reviews(productid: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get the reviews of any product on amazon by their unique id
		Example ---- ID=B01K1HPA60 (the part after /dp/)
		
		 https://www.amazon.com/Philips-Norelco-Multigroom-attachment-MG3750/dp/B01K1HPA60/?_encoding=UTF8&pd_rd_w=iQWbW&content-id=amzn1.sym.3f4ca281-e55c-46d1-9425-fb252d20366f&pf_rd_p=3f4ca281-e55c-46d1-9425-fb252d20366f&pf_rd_r=VMK4XH6AE64FWVHEZ293&pd_rd_wg=J1gHH&pd_rd_r=f80379db-7954-4243-864e-6a50b47c90be&ref_=pd_gw_exports_top_sellers_unrec"
    
    """
    url = f"https://ar7-amazon-scraper-api.p.rapidapi.com/products/{productid}/reviews"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ar7-amazon-scraper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_details(api_key: str, productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get the detail of any product from amazon by typing their id form the url above"
    
    """
    url = f"https://ar7-amazon-scraper-api.p.rapidapi.com/products/{productid}"
    querystring = {'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ar7-amazon-scraper-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

