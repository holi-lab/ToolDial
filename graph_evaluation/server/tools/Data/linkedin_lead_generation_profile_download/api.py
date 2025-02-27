import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_downloaded_profiles(page_current: int=1, downloaded_after_time: int=None, input_url: str=None, page_size: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Easily get downloaded profiles & emails:
		- fetch and paginate through all downloaded profiles
		- fetch profiles, downloaded after specified unix timestamp (useful for retrieving new profiles in batches)
		- fetch profile, by url provided in the download/profile endpoint
		
		Profiles are sorted by download time descending."
    page_current: Which page to fetch?
        downloaded_after_time: Get profiles downloaded after or at unix timestamp (greater than or equal to)
        input_url: Get profile, by URL sent in download/profile endpoint
        page_size: How many profiles per request?
        
    """
    url = f"https://linkedin-lead-generation-profile-download.p.rapidapi.com/get/profile"
    querystring = {}
    if page_current:
        querystring['page_current'] = page_current
    if downloaded_after_time:
        querystring['downloaded_after_time'] = downloaded_after_time
    if input_url:
        querystring['input_url'] = input_url
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-lead-generation-profile-download.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

