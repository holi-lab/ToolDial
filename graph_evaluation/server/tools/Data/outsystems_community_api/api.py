import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_profiles(sort_by: str=None, order_by: str=None, page: int=1, limit: int=50, keyword: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of profiles"
    sort_by: String (desc, asc). Orders the results by either Ascending or Descending order.
        order_by: String (name, rank, kudos, components, solutions, profile_score, training_score, forums_score, forge_score, ideas_score, total_score, forums_posts, forums_comments, ideas_submitted, ideas_commented, forge_components, publications_articles, certifications). Sorts the results by chosen value.
        page: Used to see the next page of profiles, eg limit=15 and page=2 will show you profiles 15-30
        limit: The limit of results per page that has been set. Integer between 1 - 50 (inclusive)
        keyword: Search keyword against profile name, job title, company or location.
        
    """
    url = f"https://outsystems-community-api.p.rapidapi.com/get_profiles"
    querystring = {}
    if sort_by:
        querystring['sort_by'] = sort_by
    if order_by:
        querystring['order_by'] = order_by
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "outsystems-community-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_profile_series(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns time series snapshots of a profile."
    
    """
    url = f"https://outsystems-community-api.p.rapidapi.com/get_profile_series"
    querystring = {'profile_id': profile_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "outsystems-community-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_profile(profile_url: str=None, profile_id: str='yjjslxnjng', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the details of a profile."
    
    """
    url = f"https://outsystems-community-api.p.rapidapi.com/get_profile"
    querystring = {}
    if profile_url:
        querystring['profile_url'] = profile_url
    if profile_id:
        querystring['profile_id'] = profile_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "outsystems-community-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

