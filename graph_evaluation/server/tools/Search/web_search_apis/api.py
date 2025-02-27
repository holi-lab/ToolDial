import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def web(q: str, setlang: str=None, offset: int=0, safesearch: str=None, cc: str=None, count: int=10, freshness: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search billions of web pages"
    setlang: language code
        offset: skip this many results
        safesearch: Off
Moderate
Strict
        cc: country code

        count: < 50
        freshness: Day
Week
Month
        
    """
    url = f"https://web-search-apis.p.rapidapi.com/web"
    querystring = {'q': q, }
    if setlang:
        querystring['setLang'] = setlang
    if offset:
        querystring['offset'] = offset
    if safesearch:
        querystring['safeSearch'] = safesearch
    if cc:
        querystring['cc'] = cc
    if count:
        querystring['count'] = count
    if freshness:
        querystring['freshness'] = freshness
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-search-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def podcasts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Click here:
		https://rapidapi.com/searchapis/api/web-podcasts
		
		150M+ podcast episodes"
    
    """
    url = f"https://web-search-apis.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-search-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Click here:
		https://rapidapi.com/searchapis/api/web-typeahead"
    
    """
    url = f"https://web-search-apis.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-search-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images(q: str, setlang: str=None, cc: str=None, color: str=None, aspect: str=None, minwidth: int=None, minheight: int=None, freshness: str=None, size: str=None, maxwidth: int=None, imagetype: str=None, count: int=10, license: str=None, maxheight: int=None, safesearch: str=None, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Original images from websites"
    setlang: language code
        cc: country code
        color: Monochrome
Black
White
Blue
Brown
Gray
Green
Orange
Pink
Purple
Red
Teal
Yellow
        aspect: Square
Tall
Wide
        freshness: Day
Week
Month
        size: Small
Medium
Large
        imagetype: AnimatedGif
Clipart
Line
Photo
Shopping
Transparent
        count: < 150
        license: All
Public
Share
ShareCommercially
Modify
ModifyCommercially
        safesearch: Off
Moderate
Strict
        offset: skip this many results
        
    """
    url = f"https://web-search-apis.p.rapidapi.com/images"
    querystring = {'q': q, }
    if setlang:
        querystring['setLang'] = setlang
    if cc:
        querystring['cc'] = cc
    if color:
        querystring['color'] = color
    if aspect:
        querystring['aspect'] = aspect
    if minwidth:
        querystring['minWidth'] = minwidth
    if minheight:
        querystring['minHeight'] = minheight
    if freshness:
        querystring['freshness'] = freshness
    if size:
        querystring['size'] = size
    if maxwidth:
        querystring['maxWidth'] = maxwidth
    if imagetype:
        querystring['imageType'] = imagetype
    if count:
        querystring['count'] = count
    if license:
        querystring['license'] = license
    if maxheight:
        querystring['maxHeight'] = maxheight
    if safesearch:
        querystring['safeSearch'] = safesearch
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-search-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(q: str, cc: str=None, setlang: str=None, freshness: str=None, offset: int=0, count: int=10, safesearch: str=None, sortby: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "News from every source"
    cc: country code
        setlang: language code
        freshness: Day
Week
Month
        offset: skip this many results
        count: < 100
        safesearch: Off
Moderate
Strict
        sortby: Date
Relevance
        
    """
    url = f"https://web-search-apis.p.rapidapi.com/news"
    querystring = {'q': q, }
    if cc:
        querystring['cc'] = cc
    if setlang:
        querystring['setLang'] = setlang
    if freshness:
        querystring['freshness'] = freshness
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if safesearch:
        querystring['safeSearch'] = safesearch
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-search-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos(q: str, cc: str=None, setlang: str=None, resolution: str=None, freshness: str=None, count: int=10, videolength: str=None, safesearch: str=None, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Videos with thumbnails"
    cc: country code
        setlang: language code
        resolution: 360p
480p
720p
1080p
        freshness: Day
Week
Month
        count: < 100
        videolength: Short
Medium
Long
        safesearch: Moderate
Strict
        offset: skip this many results
        
    """
    url = f"https://web-search-apis.p.rapidapi.com/videos"
    querystring = {'q': q, }
    if cc:
        querystring['cc'] = cc
    if setlang:
        querystring['setLang'] = setlang
    if resolution:
        querystring['resolution'] = resolution
    if freshness:
        querystring['freshness'] = freshness
    if count:
        querystring['count'] = count
    if videolength:
        querystring['videoLength'] = videolength
    if safesearch:
        querystring['safeSearch'] = safesearch
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-search-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

