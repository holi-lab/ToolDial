import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_wallpapers_by_each_category(categoryslug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For get wallpapers, you should pass the categorySlug which recieved with categories endpoint.
		Ex: /categories/bts-dp-wallpapers
		
		"bts-dp-wallpapers" is the example categorySlug which you should pass"
    
    """
    url = f"https://bts-wallpapers.p.rapidapi.com/categories/{categoryslug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bts-wallpapers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API distribute wallpapers under the various categories. So, you should get the "categoryName" & "categorySlug" first. wallpaper count of each category also included."
    
    """
    url = f"https://bts-wallpapers.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bts-wallpapers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

