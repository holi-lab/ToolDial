import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_search_by_image(url: str, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reverse search by Image between billion of images.
		
		This endpoint supports "URL" parameter only. For "image" file upload, use POST endpoint."
    url: The URL to search images by. A default URL from Wikipedia containing the following pizza has been used.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/260px-Eq_it-na_pizza-margherita_sep2005_sml.jpg)
        
    """
    url = f"https://webit-image-search.p.rapidapi.com/reverse"
    querystring = {'url': url, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-image-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similar(image_id: str, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Visually Similar images to an image from a previous Search result."
    image_id: The \"image_id\" of an image from search results to search similar images by.

The used image_id corresponds to the following image:
![](https://img.freepik.com/free-photo/3d-aesthetics-with-shapes-vaporwave-style_23-2148981118.jpg?size=262&ext=jpg)
        
    """
    url = f"https://webit-image-search.p.rapidapi.com/similar"
    querystring = {'image_id': image_id, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-image-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, number: int=10, search_filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Webit Image Search API provides you a powerful endpoint to search billions of images from the world wide web featuring rating, reverse search by image and multi-lingual capabilities."
    
    """
    url = f"https://webit-image-search.p.rapidapi.com/search"
    querystring = {'q': q, }
    if number:
        querystring['number'] = number
    if search_filter:
        querystring['search_filter'] = search_filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-image-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

