import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def creditcards(quantity: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 credit cards with en_US locale"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/credit_cards"
    querystring = {'_quantity': quantity, '_locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom(vatid: str, quantity: int, birthday: str, is_from: str, name: str, surname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 rows with first name, last name, country, birthday, email and vat"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/custom"
    querystring = {'vatId': vatid, '_quantity': quantity, 'birthday': birthday, 'from': is_from, 'name': name, 'surname': surname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places(quantity: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 places"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/places"
    querystring = {'_quantity': quantity, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def persons(locale: str, quantity: int, gender: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 persons with en_US locale and gender female"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/persons"
    querystring = {'_locale': locale, '_quantity': quantity, '_gender': gender, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users(quantity: int, gender: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 users with en_US locale and gender male"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/users"
    querystring = {'_quantity': quantity, '_gender': gender, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addresses(quantity: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 addresses with en_US locale"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/addresses"
    querystring = {'_quantity': quantity, '_locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def texts(quantity: int, locale: str, characters: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 texts with it_IT locale and 250 characters length"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/texts"
    querystring = {'_quantity': quantity, '_locale': locale, '_characters': characters, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products(categories_type: str, taxes: int, locale: str, quantity: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 products with en_US locale and taxes of 10% and categories type set as string"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/products"
    querystring = {'_categories_type': categories_type, '_taxes': taxes, '_locale': locale, '_quantity': quantity, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies(quantity: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 companies with en_US locale"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/companies"
    querystring = {'_quantity': quantity, '_locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images(height: int, locale: str, type: str, quantity: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 kitten images with height of 300 pixels and locale en_US"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/images"
    querystring = {'_height': height, '_locale': locale, '_type': type, '_quantity': quantity, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def books(quantity: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 100 books with en_US locale"
    
    """
    url = f"https://seeding-data.p.rapidapi.com/api/v1/books"
    querystring = {'_quantity': quantity, '_locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeding-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

