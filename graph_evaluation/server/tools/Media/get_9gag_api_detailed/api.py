import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_pages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pages."
    
    """
    url = f"https://9gag-api-detailed.p.rapidapi.com/get_pages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "9gag-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all categories."
    
    """
    url = f"https://9gag-api-detailed.p.rapidapi.com/get_categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "9gag-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_posts(username: str, counter: str=None, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user posts."
    counter: leave blank for first request. For subsequent requests, send the counter and after parameter you got from the previous request.
        after: leave blank for first request. For subsequent requests, send the counter and after parameter you got from the previous request.
        
    """
    url = f"https://9gag-api-detailed.p.rapidapi.com/get_user_posts"
    querystring = {'username': username, }
    if counter:
        querystring['counter'] = counter
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "9gag-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_profile(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user profie."
    
    """
    url = f"https://9gag-api-detailed.p.rapidapi.com/get_user"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "9gag-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post_data_download_video_photo(post_id: str='adPXX3Q', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns post data. Includes download link for videos."
    
    """
    url = f"https://9gag-api-detailed.p.rapidapi.com/get_post"
    querystring = {}
    if post_id:
        querystring['post_id'] = post_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "9gag-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_from_page(category: str, counter: str=None, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns page posts."
    category: Supports categories returned from the /get_pages endpoint

        counter: leave blank for first request. For subsequent requests, send the counter and after parameter you got from the previous request.

        after: leave blank for first request. For subsequent requests, send the counter and after parameter you got from the previous request.

        
    """
    url = f"https://9gag-api-detailed.p.rapidapi.com/get_page_posts"
    querystring = {'category': category, }
    if counter:
        querystring['counter'] = counter
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "9gag-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_from_category(category: str, counter: str=None, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns post from specific category. Uses cursor as **after** parameter."
    counter: leave blank for first request. For subsequent requests, send the counter and after parameter you got from the previous request.

        after: leave blank for first request. For subsequent requests, send the counter and after parameter you got from the previous request.

        
    """
    url = f"https://9gag-api-detailed.p.rapidapi.com/get_posts_from_category"
    querystring = {'category': category, }
    if counter:
        querystring['counter'] = counter
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "9gag-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post_comments(post_id: str, count: str, next: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns post comments. Uses cursor (**after **parameter)."
    next: To get the **next **data, leave the** nex**t parameter*** blank***. then send the next data in the incoming response as the next parameter and increase the count amount by 10. You can also pull other comments by doing it constantly.

        
    """
    url = f"https://9gag-api-detailed.p.rapidapi.com/get_post_comments"
    querystring = {'post_id': post_id, 'count': count, }
    if next:
        querystring['next'] = next
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "9gag-api-detailed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

