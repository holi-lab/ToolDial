import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_my_available_offers_copy(iso6523code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get My Available Offers based on iso6523Code"
    
    """
    url = f"https://batra-link.p.rapidapi.com/GetMyAvailableOffer/{iso6523code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def update_stock(iso6523code: str, customer_iso6523code: str, new_inventorylevel: str, gtin: str, validfrom: str, batchnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Update the stock of an offer in Croaaaa"
    
    """
    url = f"https://batra-link.p.rapidapi.com/Croaaaa/UpdateStock/{iso6523code}/{gtin}/{batchnumber}/{validfrom}/{customer_iso6523code}/{new_inventorylevel}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delete_offer(gtin: str, customer_iso6523code: str, validfrom: str, batchnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete a offer from Croaaaa"
    
    """
    url = f"https://batra-link.p.rapidapi.com/Croaaaa/DeleteOffer/{iso6523code}/{gtin}/{batchnumber}/{validfrom}/{customer_iso6523code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_my_available_offers(iso6523code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get My Available Offers based on iso6523Code"
    
    """
    url = f"https://batra-link.p.rapidapi.com/Croaaaa/GetMyAvailableOffer/{iso6523code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_my_offers(iso6523code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get My Offers based on iso6523Code"
    
    """
    url = f"https://batra-link.p.rapidapi.com/Croaaaa/GetMyOffer/{iso6523code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_offer(gtin: str, validfrom: str, customer_iso6523code: str, pricecurrency: str, price: str, validthrough: str, inventorylevel_code: str, productiondate: str, batchnumber: str, inventorylevel_value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a offer in Croaaaa"
    
    """
    url = f"https://batra-link.p.rapidapi.com/Croaaaa/CreateOffer/{iso6523code}/{gtin}/{price}/{pricecurrency}/{batchnumber}/{productiondate}/{inventorylevel_value}/{inventorylevel_code}/{validfrom}/{validthrough}/{customer_iso6523code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_organization_by_iso6523code(iso6523code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Organization By iso6523Code in format Schema.org"
    
    """
    url = f"https://batra-link.p.rapidapi.com/Batra/020/{iso6523code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_by_gtin_copy(gtin: str, iso6523code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Product By GTIN in format GS1 Voc 1.7"
    
    """
    url = f"https://batra-link.p.rapidapi.com/Batra/020/{iso6523code}"
    querystring = {'gtin': gtin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_by_gtin(gtin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Product By GTIN in format GS1 Voc 1.7"
    
    """
    url = f"https://batra-link.p.rapidapi.com/Batra/01/{gtin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_products_datas_based_on_gtin(gtin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get products datas based on GTIN"
    
    """
    url = f"https://batra-link.p.rapidapi.com/01/{gtin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_colruyt_offers_by_gtin(gtin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Colruyt Halle history offers by GTIN"
    
    """
    url = f"https://batra-link.p.rapidapi.com/inflacheck/colruyt/{gtin}"
    querystring = {}
    if gtin:
        querystring['gtin'] = gtin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_bio_planet_offers_by_gtin(gtin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Bio-Planet offers history by GTIN"
    
    """
    url = f"https://batra-link.p.rapidapi.com/inflacheck/bioplanet/{gtin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_carrefour_offers_by_gtin(gtin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Carrefour offers history by GTIN"
    
    """
    url = f"https://batra-link.p.rapidapi.com/inflacheck/carrefour/{gtin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_products_datas_based_on_gln(gln: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get products datas based on GLN"
    
    """
    url = f"https://batra-link.p.rapidapi.com/020/{gln}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "batra-link.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

