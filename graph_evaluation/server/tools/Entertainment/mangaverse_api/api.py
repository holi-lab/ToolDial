import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_images(is_id: str='6486b7937ae7cb74845df888', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will be get the image from the chapters, the image sort base on the index
		so make sure you sort base from index if you manipulate it,
		id should be chapter id"
    
    """
    url = f"https://mangaverse-api.p.rapidapi.com/manga/image"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangaverse-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_chapters(is_id: str='6486b7547ae7cb74845df856', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get all the chapters list from the manga 
		id should be manga id"
    
    """
    url = f"https://mangaverse-api.p.rapidapi.com/manga/chapter"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangaverse-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_manga(is_id: str='647fed35c71c2c9122b318f8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manga details 
		id should be manga id"
    
    """
    url = f"https://mangaverse-api.p.rapidapi.com/manga"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangaverse-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_latest(genres: str='Harem,Fantasy', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get the latest of the updated manga on this server
		page should be 1 or more than 1 and the genres should be string with comma
		genres ex: Harem,Fantasy"
    
    """
    url = f"https://mangaverse-api.p.rapidapi.com/manga/latest"
    querystring = {}
    if genres:
        querystring['genres'] = genres
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangaverse-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_manga(text: str='isekai', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch manga base on the text, text will be include from title and sub title"
    
    """
    url = f"https://mangaverse-api.p.rapidapi.com/manga/search"
    querystring = {}
    if text:
        querystring['text'] = text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangaverse-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_manga(page: str='1', genres: str='Harem,Fantasy', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will get the list of manga on this server
		page should be 1 or more than 1 and the genres should be string with comma
		genres ex: Harem,Fantasy"
    
    """
    url = f"https://mangaverse-api.p.rapidapi.com/manga/fetch"
    querystring = {}
    if page:
        querystring['page'] = page
    if genres:
        querystring['genres'] = genres
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mangaverse-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

