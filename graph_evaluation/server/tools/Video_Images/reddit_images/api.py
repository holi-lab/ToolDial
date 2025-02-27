import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_image_by_category(category: str, uuid: str='202110-2612-3859-', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 50 images from a specific category. To get the next 50 images from the same category, add as a query the fields of  "LastEvaluatedKey" which are category and uuid. By using the LastEvaluatedKey you can browsee through the pages of images. To get the categories please use the other"
    
    """
    url = f"https://reddit-images.p.rapidapi.com/{category}"
    querystring = {}
    if uuid:
        querystring['uuid'] = uuid
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-images.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_available_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint to receive an array of String with the name of all the categories avaible with images."
    
    """
    url = f"https://reddit-images.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-images.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

