import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def website_icon(site: str, platform: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the favicon or favicon-related icon of a website: its classic favicon, its touch icon, its icon for Android, etc."
    site: The URL of the website to analyze
        platform: The targeted platform. For example, you may want to get the icon used by iOS Safari (ie. the Apple Touch icon). In this case, use ios. Possible values are ios, android_chrome, windows, safari_pinned_tab and desktop (the default value)
        
    """
    url = f"https://realfavicongenerator.p.rapidapi.com/favicon/icon"
    querystring = {'site': site, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realfavicongenerator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def website_preview(site: str, platform: str='desktop', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a preview of the website for a given platform. For example, by passing ios as the platform, the API sends an image representing how the site would look like if it was added to the home screen of an iOS device."
    site: The URL of the website
        platform: The targeted platform. Possible values are desktop, ios, android_chrome, windows and safari_pinned_tab
        
    """
    url = f"https://realfavicongenerator.p.rapidapi.com/favicon/preview"
    querystring = {'site': site, }
    if platform:
        querystring['platform'] = platform
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realfavicongenerator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def website_analysis(site: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Analyze the favicon and related icons of a website."
    site: The URL of the website to analyze
        
    """
    url = f"https://realfavicongenerator.p.rapidapi.com/favicon/analysis"
    querystring = {'site': site, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realfavicongenerator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

