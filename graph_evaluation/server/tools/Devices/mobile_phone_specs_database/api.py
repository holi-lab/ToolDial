import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_phone_image_ids_by_custom_id(phonecustomid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get up to 10 phone images (IDs). 
		Use these IDs in the Endpoint:  Get {Image} by {Phone Image id}
		The image IDs expire after 30 minutes."
    
    """
    url = f"https://mobile-phone-specs-database.p.rapidapi.com/gsm/get-phone-images-ids-by-phone-custom-id/{phonecustomid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-phone-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specifications_by_brand_name_and_model_name(brandname: str, modelname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all phone specifications by brand name and model name"
    
    """
    url = f"https://mobile-phone-specs-database.p.rapidapi.com/gsm/get-specifications-by-brandname-modelname/{brandname}/{modelname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-phone-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_image_by_phone_image_id(imageid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- Get Image by Id
		- Paste image ID from this api "Get all phone image ids by phone custom id" and image media content is returned
		- ex of image id : "Z2RjOG80Smk2WVV2R0wwR1F1RjRNZG5TR1ZqUUZuU1JhU1pNT0FjNG94UT0=""
    
    """
    url = f"https://mobile-phone-specs-database.p.rapidapi.com/gsm/get-image-by-id/{imageid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-phone-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specifications_by_custom_id(phonecustomid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all phone specifications by phone custom id"
    
    """
    url = f"https://mobile-phone-specs-database.p.rapidapi.com/gsm/get-specifications-by-phone-custom-id/{phonecustomid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-phone-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_models_by_phone_brand(brandname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List Models By Phone Brand name"
    
    """
    url = f"https://mobile-phone-specs-database.p.rapidapi.com/gsm/get-models-by-brandname/{brandname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-phone-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_phone_brands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1. List all Phone Brands"
    
    """
    url = f"https://mobile-phone-specs-database.p.rapidapi.com/gsm/all-brands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-phone-specs-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

