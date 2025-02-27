import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getallcategories(format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    format: json or xml or php
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcategoryinfo(category: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    category: The name of the category.
        format: json or xml or php
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'category': category, 'format': format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrelatedvideos(category: str, format: str, page: int=None, per_page: int=None, summary_response: bool=None, full_response: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    category: The name of the category.
        format: json or xml or php
        page: The page number to show.
        per_page: Number of items to show on each page. Max 50.
        summary_response: Set this parameter to get back more information.
        full_response: Set this parameter to get back the full video information.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'category': category, 'format': format, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if summary_response:
        querystring['summary_response'] = summary_response
    if full_response:
        querystring['full_response'] = full_response
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrelatedchannels(category: str, format: str, page: int=None, per_page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    category: Category name
        format: json or xml or php
        page: The page number to show.
        per_page: Number of items to show on each page. Max 50.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'category': category, 'format': format, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvideofromuser(format: str, user_id: str, sort: str=None, page: int=None, per_page: str=None, summary_response: bool=None, full_response: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all videos credited to a user (both uploaded and appears)."
    format: json/xml/php
        user_id: The ID number or username of the user. A token may be used instead.
        sort: Method to sort by: newest, oldest, most_played, most_commented, or most_liked.
        page: The page number to show.
        per_page: Number of items to show on each page. Max 50.
        summary_response: Set this parameter to get back more information.
        full_response: Set this parameter to get back the full video information.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'format': format, 'user_id': user_id, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if summary_response:
        querystring['summary_response'] = summary_response
    if full_response:
        querystring['full_response'] = full_response
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvideosbytag(format: str, tag: str, page: int=None, per_page: str=None, summary_response: bool=None, full_response: bool=None, sort: str='most_commented', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of videos that have the specified tag."
    format: json/php/xml
        tag: The tag to get
        page: Page number to show
        per_page: Number of items to show on each page. Max 50.
        summary_response: Set this parameter to get back more information.
        full_response: Set this parameter to get back the full video information.
        sort: Method to sort by: relevant, newest, oldest, most_played, most_commented, or most_liked.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'format': format, 'tag': tag, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if summary_response:
        querystring['summary_response'] = summary_response
    if full_response:
        querystring['full_response'] = full_response
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallchannels(format: str, per_page: str, sort: str='most_recently_updated', page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all public channels."
    format: json/xml/php
        per_page: Number of items to show on each page. Max 50.
        sort: Method to sort by: newest, oldest, alphabetical, most_videos, most_subscribed, or most_recently_updated.
        page: The page number to show.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'format': format, 'per_page': per_page, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchannelvideos(format: str, channel_id: str, full_response: bool, user_id: str=None, page: int=None, per_page: int=None, summary_response: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the videos in a channel."
    format: json/xml/php
        channel_id: The numeric id of the channel or its url name.
        full_response: Set this parameter to get back the full video information
        user_id: The ID number or username of the user. A token may be used instead.
        page: The page number to show.
        summary_response: Set this parameter to get back more information.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'format': format, 'channel_id': channel_id, 'full_response': full_response, }
    if user_id:
        querystring['user_id'] = user_id
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if summary_response:
        querystring['summary_response'] = summary_response
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrelatedtags(category: str, format: str, page: int=None, per_page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of related tags for a category."
    category: The Name of category
        format: json or xml or php
        page: The page number to show
        per_page: Number of items to show on each page. Max 50.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'category': category, 'format': format, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchvideos(format: str, query: str, user_id: str=None, page: int=None, per_page: int=None, summary_response: bool=None, full_response: bool=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for videos."
    format: json/xml/php
        query: The search terms
        user_id: The ID number or username of the user.
        page: The page number to show.
        per_page: Number of items to show on each page. Max 50.
        summary_response: Set this parameter to get back more information.
        full_response: Set this parameter to get back the full video information.
        sort: Method to sort by: relevant, newest, oldest, most_played, most_commented, or most_liked.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'format': format, 'query': query, }
    if user_id:
        querystring['user_id'] = user_id
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if summary_response:
        querystring['summary_response'] = summary_response
    if full_response:
        querystring['full_response'] = full_response
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchannelinfo(format: str, channel_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the information on a single channel."
    format: json/xml/php
        channel_id: The numeric id of the channel or its url name.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'format': format, 'channel_id': channel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrelatedpeople(category: str, format: str, page: int=None, per_page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of related people for a category."
    category: The name of the category.
        format: json or xml or php
        page: The page number to show.
        per_page: Number of items to show on each page. Max 50.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'category': category, 'format': format, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvideoscomment(format: str, video_id: str, page: int=None, per_page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the comments on a video."
    format: json/php/xml
        video_id: The ID of the video.
        page: The page number to show.
        per_page: Number of items to show on each page. Max 50.
        
    """
    url = f"https://vimeo.p.rapidapi.com/v2"
    querystring = {'format': format, 'video_id': video_id, }
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vimeo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

