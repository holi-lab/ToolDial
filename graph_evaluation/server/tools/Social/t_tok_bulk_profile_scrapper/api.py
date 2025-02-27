import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def profile(username: str, schematype: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "T-tok Profile by Username"
    schematype: schemaType=1 Default JSON
schemaType=2 Custom JSON
        
    """
    url = f"https://t-tok-bulk-profile-scrapper.p.rapidapi.com/profile/{username}"
    querystring = {}
    if schematype:
        querystring['schemaType'] = schematype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t-tok-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds(username: str, schematype: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Feeds by Username"
    schematype: schemaType=1 Default JSON
schemaType=2 Custom JSON
        
    """
    url = f"https://t-tok-bulk-profile-scrapper.p.rapidapi.com/feeds/{username}"
    querystring = {}
    if schematype:
        querystring['schemaType'] = schematype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t-tok-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_by_secuid(username: str, schematype: str='1', secuid: str='MS4wLjABAAAAM3R2BtjzVT-uAtstkl2iugMzC6AtnpkojJbjiOdDDrdsTiTR75-8lyWJCY5VvDrZ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch feeds from secUid"
    
    """
    url = f"https://t-tok-bulk-profile-scrapper.p.rapidapi.com/feeds/{username}"
    querystring = {}
    if schematype:
        querystring['schemaType'] = schematype
    if secuid:
        querystring['secUid'] = secuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t-tok-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_by_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch video data from TikToc video url 
		
		e.g. 
		https://www.****.com/@therock/video/6800111723257941253
		(copy exact url from browser)"
    
    """
    url = f"https://t-tok-bulk-profile-scrapper.p.rapidapi.com/video"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "t-tok-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

