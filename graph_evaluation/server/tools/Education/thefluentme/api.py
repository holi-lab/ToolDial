import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_post_by_id(post_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a post by `post_id`. You can get the `post_id` from the `Get all posts` endpoint."
    
    """
    url = f"https://thefluentme.p.rapidapi.com/post/{post_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thefluentme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_posts(per_page: int=10, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of all posts.
		
		Please provide the following parameters in the query string
		- **page**: The page number. If no value is provided, the default value of `1` will be applied.
		- **per_page**: Posts per page. If no value is provided, the default value of `10` will be applied."
    
    """
    url = f"https://thefluentme.p.rapidapi.com/post"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thefluentme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all supported languages. The API supports 40 different languages and 80 different voices for the ai_reading. This is used to define the post language when a post is added. It also defines the voice for the sound file that is generated for the user to listen to."
    
    """
    url = f"https://thefluentme.p.rapidapi.com/language"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thefluentme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_language_by_id(language_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a language by id"
    
    """
    url = f"https://thefluentme.p.rapidapi.com/language/{language_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thefluentme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_score_by_id(score_id: str, scale: int=90, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a score by `score_id`. You can get the `score_id` from the `Get all scores` endpoint. The scale parameter defines the maximum number of points for the recording, as well as each word. This parameter is not required. The default value of 100 will be applied if the parameter is not provided."
    
    """
    url = f"https://thefluentme.p.rapidapi.com/score/{score_id}"
    querystring = {}
    if scale:
        querystring['scale'] = scale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thefluentme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_scores(page: int=1, per_page: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of all scores.
		
		Please provide the following parameters in the query string
		- **page**: The page number. If no value is provided, the default value of `1` will be applied.
		- **per_page**: Scores per page. If no value is provided, the default value of `10` will be applied."
    
    """
    url = f"https://thefluentme.p.rapidapi.com/score"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thefluentme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

