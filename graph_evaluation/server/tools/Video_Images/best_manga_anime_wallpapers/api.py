import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_sub_categories_relavant_to_specific_category(category_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## About this endpoint
		If you want to get filtered response of items which relavant to specific category, you can pass the category name as path paramenter.
		
		Example : For get all wallpapers which in "Popular" category,
		Request this endpoint, /categories/popular
		
		/popular is the path parameter in this endpoint. you can pass other categorie's name as that."
    
    """
    url = f"https://best-manga-anime-wallpapers.p.rapidapi.com/categories/{category_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-manga-anime-wallpapers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_wallpapers(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## About this Endpoint
		For get Wallpapers, Pass the slug value sub category (you want to get wallpapers) as the Path parameter,
		
		### Example : to get all wallpapers of Attack on Titan
		Request ***/wallpapers/attack-on-titan-hd-wallpapers***"
    
    """
    url = f"https://best-manga-anime-wallpapers.p.rapidapi.com/wallpapers/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-manga-anime-wallpapers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Structure of Response
		This response contains array of categories and each one have category name and wallpapers count which relavant to category."
    
    """
    url = f"https://best-manga-anime-wallpapers.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-manga-anime-wallpapers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_popular_anime_characters_manga_anime_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Structure of Response
		Response contains 3 Arrays of subcategories (Popular, Characters, Anime)
		
		Each JSON Response item contains fields, "thumbnail, title, wallpaper_count, slug, category""
    
    """
    url = f"https://best-manga-anime-wallpapers.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-manga-anime-wallpapers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

