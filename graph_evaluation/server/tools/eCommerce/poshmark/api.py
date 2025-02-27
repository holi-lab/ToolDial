import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, accept_encoding: str='gzip, deflate', domain: str='com', page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a listing."
    domain: com, ca, in, com.au
        page: Pagination cursor. Get from a response at `next_page` field.
        
    """
    url = f"https://poshmark.p.rapidapi.com/search"
    querystring = {'query': query, }
    if accept_encoding:
        querystring['Accept-Encoding'] = accept_encoding
    if domain:
        querystring['domain'] = domain
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "poshmark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def following_tab(username: str='thebandteeshop', next_max_id: int=None, url: str=None, domain: str='com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Following Tab"
    next_max_id: Use it for pagination.
        url: `https://poshmark.ca/user/thebandteeshop/shares`
        domain: com, ca, in, com.au
        
    """
    url = f"https://poshmark.p.rapidapi.com/following"
    querystring = {}
    if username:
        querystring['username'] = username
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    if url:
        querystring['url'] = url
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "poshmark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers_tab(url: str=None, next_max_id: int=None, domain: str='com', username: str='thebandteeshop', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Followers Tab"
    url: `https://poshmark.ca/user/thebandteeshop/shares`
        next_max_id: Use it for pagination.
        domain: com, ca, in, com.au
        
    """
    url = f"https://poshmark.p.rapidapi.com/followers"
    querystring = {}
    if url:
        querystring['url'] = url
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    if domain:
        querystring['domain'] = domain
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "poshmark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shares_tab(next_max_id: int=None, username: str='thebandteeshop', domain: str='com', url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Shares Tab"
    next_max_id: Use it for pagination.
        domain: com, ca, in, com.au
        url: `https://poshmark.ca/user/thebandteeshop/shares`
        
    """
    url = f"https://poshmark.p.rapidapi.com/shares"
    querystring = {}
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    if username:
        querystring['username'] = username
    if domain:
        querystring['domain'] = domain
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "poshmark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details_about_the_user(username: str, domain: str='com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details about the User."
    username: Get details about the User.
See response example for details!
        domain: com, ca, in, com.au
        
    """
    url = f"https://poshmark.p.rapidapi.com/user"
    querystring = {'username': username, }
    if domain:
        querystring['domain'] = domain
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "poshmark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def details_about_listing(domain: str='com', url: str=None, is_id: str='600f17e6bcdb2f92469ce851', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Details about listing by id or URL."
    domain: com, ca, in, com.au
        url: `https://poshmark.com/listing/Vintage-Levis-Custom-Made-Orange-Tab-ReDone-Jean-5b7f07e33c98440e93f90618`
        is_id: Listing id, you can get from `/closet` or from `URL`

`https://poshmark.com/listing/Nike-Air-Max-Flyknit-Zoom-Gray-Sneakers-Size-8-600f17e6bcdb2f92469ce851` => `600f17e6bcdb2f92469ce851`
        
    """
    url = f"https://poshmark.p.rapidapi.com/listing"
    querystring = {}
    if domain:
        querystring['domain'] = domain
    if url:
        querystring['url'] = url
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "poshmark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listings_closet(url: str=None, domain: str='com', username: str='atlanticfinds', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Listings (Closet) by username or url"
    url: `https://poshmark.com/closet/atlanticfinds`
        domain: com, ca, in, com.au
        page: Pagination cursor. Get from a response at `next_page` field.
        
    """
    url = f"https://poshmark.p.rapidapi.com/closet"
    querystring = {}
    if url:
        querystring['url'] = url
    if domain:
        querystring['domain'] = domain
    if username:
        querystring['username'] = username
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "poshmark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

