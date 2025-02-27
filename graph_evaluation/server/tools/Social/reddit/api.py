import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def popular_posts(sort: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Popular Posts"
    sort: you can just send `new `or `hot`
        
    """
    url = f"https://reddit34.p.rapidapi.com/getPopularPosts"
    querystring = {'sort': sort, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_comments_by_username(time: str, username: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Comments By Username"
    time: you can just select one item from below:
`hour`
`day`
`week`
`month`
`year`
`all`
        
    """
    url = f"https://reddit34.p.rapidapi.com/getTopCommentsByUsername"
    querystring = {'time': time, 'username': username, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts_by_subreddit(subreddit: str, sort: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Posts By Subreddit"
    subreddit: example: reddit.com/r/`memes`
        sort: you can just send `new `or `hot`
        
    """
    url = f"https://reddit34.p.rapidapi.com/getPostsBySubreddit"
    querystring = {'subreddit': subreddit, 'sort': sort, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_posts_by_subreddit(time: str, subreddit: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Posts By Subreddit"
    time: you can just select one item from below:
`hour`
`day`
`week`
`month`
`year`
`all`
        subreddit: example: reddit.com/r/`memes`
        
    """
    url = f"https://reddit34.p.rapidapi.com/getTopPostsBySubreddit"
    querystring = {'time': time, 'subreddit': subreddit, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_details(post_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Post Details"
    
    """
    url = f"https://reddit34.p.rapidapi.com/getPostDetails"
    querystring = {'post_id': post_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_by_username(sort: str, username: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Comments By Username"
    sort: you can just send `new `or `hot`
        
    """
    url = f"https://reddit34.p.rapidapi.com/getCommentsByUsername"
    querystring = {'sort': sort, 'username': username, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_posts_by_username(username: str, time: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Posts By Username"
    time: you can just select one item from below:
`hour`
`day`
`week`
`month`
`year`
`all`
        
    """
    url = f"https://reddit34.p.rapidapi.com/getTopPostsByUsername"
    querystring = {'username': username, 'time': time, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts_by_username(username: str, sort: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Posts By Username"
    sort: you can just send `new `or `hot`
        
    """
    url = f"https://reddit34.p.rapidapi.com/getPostsByUsername"
    querystring = {'username': username, 'sort': sort, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rising_popular_posts(cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rising Popular Posts"
    
    """
    url = f"https://reddit34.p.rapidapi.com/getRisingPopularPosts"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_popular_posts(time: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Popular Posts"
    time: you can just select one item from below:
`hour`
`day`
`week`
`month`
`year`
`all`
        
    """
    url = f"https://reddit34.p.rapidapi.com/getTopPopularPosts"
    querystring = {'time': time, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit34.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

