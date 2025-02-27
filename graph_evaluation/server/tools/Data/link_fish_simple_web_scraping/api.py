import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def urlsdata(url: str, simplify_special_types: bool=None, item_format: str='normal', browser_render: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract data"
    url: The URL of the website to query
        simplify_special_types: Some types like "PropertyValue" do save key and value in separate properties which makes the data harder to process. If this option gets set it converts them automatically into the regular key -> value format.
        item_format: If the items should be return "normal" with multiple levels or "flat" with just one level and linked instead.
        browser_render: If the page should be fully rendered with a browser to extract data. The request will then cost 5 credits instead of 1!
        
    """
    url = f"https://linkfish.p.rapidapi.com/Urls/data"
    querystring = {'url': url, }
    if simplify_special_types:
        querystring['simplify_special_types'] = simplify_special_types
    if item_format:
        querystring['item_format'] = item_format
    if browser_render:
        querystring['browser_render'] = browser_render
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkfish.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def urlsapps(url: str, browser_render: bool=None, return_urls: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get mobile apps"
    url: The URL of the website to query
        browser_render: If the page should be fully rendered with a browser to extract data. The request will then cost 5 credits instead of 1!
        return_urls: Returns app URLs instead of the identifiers
        
    """
    url = f"https://linkfish.p.rapidapi.com/Urls/apps"
    querystring = {'url': url, }
    if browser_render:
        querystring['browser_render'] = browser_render
    if return_urls:
        querystring['return_urls'] = return_urls
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkfish.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def urlsbrowserscreenshot(url: str, width: int=640, type: str='normal', file_format: str='png', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate screenshot (browser). This request will cost 5 credits instead of 1!"
    url: The URL of the website to create screenshot of
        width: The widh of the screenshot in pixel.
        type: What kind of screenshot should be returned. If it should be a regular 16:9 screenshot or one with the full page height
        file_format: The file format of the screenshot
        
    """
    url = f"https://linkfish.p.rapidapi.com/Urls/browser-screenshot"
    querystring = {'url': url, }
    if width:
        querystring['width'] = width
    if type:
        querystring['type'] = type
    if file_format:
        querystring['file_format'] = file_format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkfish.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def urlssocialmedia(url: str, browser_render: bool=None, return_urls: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get social media accounts"
    url: The URL of the website to query
        browser_render: If the page should be fully rendered with a browser to extract data. The request will then cost 5 credits instead of 1!
        return_urls: Returns profile URLs instead of the profile names/ids
        
    """
    url = f"https://linkfish.p.rapidapi.com/Urls/social-media"
    querystring = {'url': url, }
    if browser_render:
        querystring['browser_render'] = browser_render
    if return_urls:
        querystring['return_urls'] = return_urls
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkfish.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def urlsgeocoordinates(url: str, browser_render: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get geo coordinates"
    url: The URL of the website to query
        browser_render: If the page should be fully rendered with a browser to extract data. The request will then cost 5 credits instead of 1!
        
    """
    url = f"https://linkfish.p.rapidapi.com/Urls/geo-coordinates"
    querystring = {'url': url, }
    if browser_render:
        querystring['browser_render'] = browser_render
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkfish.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def urlsbrowserdata(url: str, simplify_special_types: bool=None, screenshot_width: int=640, screenshot_file_format: str='png', screenshot: str='none', item_format: str='normal', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract data (browser). This request will cost 5 credits instead of 1!"
    url: The URL of the website to query
        simplify_special_types: Some types like "PropertyValue" do save key and value in separate properties which makes the data harder to process. If this option gets set it converts them automatically into the regular key -> value format.
        screenshot_width: The widh of the screenshot in pixel.
        screenshot_file_format: The file format of the screenshot
        screenshot: If and what kind of screenshot should be returned. Do only request screenshot generation when really needed because it will increase the response time significantly.
        item_format: If the items should be return "normal" with multiple levels or "flat" with just one level and linked instead.
        
    """
    url = f"https://linkfish.p.rapidapi.com/Urls/browser-data"
    querystring = {'url': url, }
    if simplify_special_types:
        querystring['simplify_special_types'] = simplify_special_types
    if screenshot_width:
        querystring['screenshot_width'] = screenshot_width
    if screenshot_file_format:
        querystring['screenshot_file_format'] = screenshot_file_format
    if screenshot:
        querystring['screenshot'] = screenshot
    if item_format:
        querystring['item_format'] = item_format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkfish.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

