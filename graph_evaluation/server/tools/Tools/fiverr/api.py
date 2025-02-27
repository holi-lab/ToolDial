import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user_reviews(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user reviews."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/user/reviews"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_sheet(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user sheet."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/user/sheet"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_gigs(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user gigs."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/user/gigs"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user by id."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/user/page"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all categories."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/categories/tree"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recommended_gigs(gig_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recommended gigs."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/gig/recommended_gigs"
    querystring = {'gig_id': gig_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_gig(gig_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get gig by id."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/gig/page"
    querystring = {'gig_id': gig_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(query: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search users."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/search/users"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_gigs(query: str, show_pro_gigs_first: str='false', gig_price_range_max: str='5000', gigs_from_online_sellers: str='false', gig_price_range_min: str='5', category_id: str=None, sub_category_id: str=None, delivery_time: str='any', sort_type: str='auto', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search gigs with advanced filters."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/search/gigs"
    querystring = {'query': query, }
    if show_pro_gigs_first:
        querystring['show_pro_gigs_first'] = show_pro_gigs_first
    if gig_price_range_max:
        querystring['gig_price_range_max'] = gig_price_range_max
    if gigs_from_online_sellers:
        querystring['gigs_from_online_sellers'] = gigs_from_online_sellers
    if gig_price_range_min:
        querystring['gig_price_range_min'] = gig_price_range_min
    if category_id:
        querystring['category_id'] = category_id
    if sub_category_id:
        querystring['sub_category_id'] = sub_category_id
    if delivery_time:
        querystring['delivery_time'] = delivery_time
    if sort_type:
        querystring['sort_type'] = sort_type
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_suggestions(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search suggestions."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/search/suggestions"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category_gigs(sub_category_id: str, show_pro_gigs_first: str='false', sort_type: str='auto', gigs_from_online_sellers: str='false', gig_price_range_max: str='5000', gig_price_range_min: int=5, delivery_time: str='any', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get category gigs."
    
    """
    url = f"https://fiverr4.p.rapidapi.com/category/gigs"
    querystring = {'sub_category_id': sub_category_id, }
    if show_pro_gigs_first:
        querystring['show_pro_gigs_first'] = show_pro_gigs_first
    if sort_type:
        querystring['sort_type'] = sort_type
    if gigs_from_online_sellers:
        querystring['gigs_from_online_sellers'] = gigs_from_online_sellers
    if gig_price_range_max:
        querystring['gig_price_range_max'] = gig_price_range_max
    if gig_price_range_min:
        querystring['gig_price_range_min'] = gig_price_range_min
    if delivery_time:
        querystring['delivery_time'] = delivery_time
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fiverr4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

