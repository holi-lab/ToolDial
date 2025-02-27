import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_collection_by_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each collection have a unique ID. These ID can be used to get the collection"
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/collections/639caf9a9ba6cf29e8b8c221"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories_by_collection(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each collection has their own categories, You can check the collection and category endpoints to get more information. Each hadiths is under a category and each category is under a collection. 
		Example:
		{COLLECTION} ---> {CATEGORY} ---> {HADITHS}
		{Sahih Bukhari} ---> {Revelation}  --->{hadiths}
		You can seperate the categories by their specific collections.
		By default limit is 10 and page is 1. the limit and page can be altered to suit your taste."
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/categories?collection=Sahih%20Bukhari"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all categories from the database, by default limit is 10 and page is 1. the limit and page can be altered to suit your taste."
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_category_by_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each category have a unique ID. These ID can be used to get the category"
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/categories/639caf9a9ba6cf29e8b8c221"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_collections(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all collections from the database, by default limit is 10 and page is 1. the limit and page can be altered to suit your taste."
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/collections"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_hadiths_with_key_words(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searching for hadiths. Just enter the keyword, example searching for food, the request will look like this https://hadiths-api.vercel.app/api/hadiths?search=food
		
		And limit can be added to reduce the amount of data received, by default the limit is 10.
		Page number by default is page1, but it can be changed as you want.
		NB: The %20 in the example query is just an empty space as the name of the collection is Sahih Bukhari in the database."
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/hadiths?search=food"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hadiths_by_collection(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each collection has their own categories, You can check the collection and category endpoints to get more information. Each hadiths is under a category and each category is under a collection. 
		Example:
		{COLLECTION} ---> {CATEGORY} ---> {HADITHS}
		{Sahih Bukhari} ---> {Revelation}  --->{hadiths}
		You can seperate the hadiths by their specific collections.
		By default limit is 10 and page is 1. the limit and page can be altered to suit your taste."
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/hadiths?collection=Sahih%20Bukhari"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hadiths_based_on_the_category(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each h collection has their own categories, You can check the collection and category endpoints to get more information. Each hadiths is under a category and each category is under a collection. 
		Example:
		{COLLECTION} ---> {CATEGORY} ---> {HADITHS}
		{Sahih Bukhari} ---> {Revelation}  --->{hadiths}
		You can seperate the hadiths by their specific categories.
		By default limit is 10 and page is 1. the limit and page can be altered to suit your taste."
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/hadiths?collection=Sahih%20Bukhari&category=Revelation"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_hadiths_in_collection_with_key_words(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searching for hadiths based on their various collections. Just enter the keyword, example searching for food, the request will look like this https://hadiths-api.vercel.app/api/hadiths?collection=Sahih Bukhari&search=food
		
		And limit can be added to reduce the amount of data received, by default the limit is 10.
		Page number by default is page1, but it can be changed as you want.
		NB: The %20 in the example query is just an empty space as the name of the collection is Sahih Bukhari in the database."
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/hadiths?collection=Sahih%20Bukhari&search=food"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_hadiths(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all hadiths from the database, by default limit is 10 and page is 1. the limit and page can be altered to suit your taste."
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/hadiths"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hadith_by_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each hadith have a unique ID. These ID can be used to get the hadith"
    
    """
    url = f"https://hadiths-api.p.rapidapi.com/hadiths/639b1ed040467f67664485cb"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hadiths-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

