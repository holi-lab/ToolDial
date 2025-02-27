import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pornstar_search(query: str, responseimagesbase64: int=1, responseimages: int=1, responseprofileimagebase64: int=1, responseprofileimage: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a porn star using the "Query".
		A list of all stars found is returned, or a detail pornstar if only one is found."
    
    """
    url = f"https://quality-porn.p.rapidapi.com/pornstar/search"
    querystring = {'query': query, }
    if responseimagesbase64:
        querystring['responseImagesBase64'] = responseimagesbase64
    if responseimages:
        querystring['responseImages'] = responseimages
    if responseprofileimagebase64:
        querystring['responseProfileImageBase64'] = responseprofileimagebase64
    if responseprofileimage:
        querystring['responseProfileImage'] = responseprofileimage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pornstar_by_name(name: str, responseprofileimagebase64: int=1, responseimagesbase64: int=1, responseimages: int=1, responseprofileimage: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Loads a porn star by name from the loaded list."
    
    """
    url = f"https://quality-porn.p.rapidapi.com/pornstar/detail"
    querystring = {'name': name, }
    if responseprofileimagebase64:
        querystring['responseProfileImageBase64'] = responseprofileimagebase64
    if responseimagesbase64:
        querystring['responseImagesBase64'] = responseimagesbase64
    if responseimages:
        querystring['responseImages'] = responseimages
    if responseprofileimage:
        querystring['responseProfileImage'] = responseprofileimage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def docs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available routs and parameters as json."
    
    """
    url = f"https://quality-porn.p.rapidapi.com/docs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supported_sites(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available pages with the possible qualities and filters."
    
    """
    url = f"https://quality-porn.p.rapidapi.com/supported-sites"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_search(query: str, site: str=None, timeout: int=5000, page: int=1, filter: str=None, quality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for quality porn and get a provider-subdivided list of the best matching videos.
		
		### Examples
		
		Search only HD videos that have been rated best with the search term "amateur".
		`/search?quality=hd&filter=rating&query=amateur'`
		
		Search for "amateur" only. Filters and quality are optional.
		`/search?query=amateur'`
		
		Search for "amateur" on page 2.
		`/search?query=amateur&page=2'`
		
		Search for "amateur" and set a timeout for 5 seconds for each site request.
		`/search?query=amateur&timeout=5000`
		
		Filter only by the page you need. Here for example by PornHub. It can be filtered by any name part or part URL. For example "porn", ".com" or complete names "pornhub", "youporn.com".
		`/search?query=amateur&site=pornhub`"
    filter: Filter the site results.
        quality: Set minimum quality
        
    """
    url = f"https://quality-porn.p.rapidapi.com/search"
    querystring = {'query': query, }
    if site:
        querystring['site'] = site
    if timeout:
        querystring['timeout'] = timeout
    if page:
        querystring['page'] = page
    if filter:
        querystring['filter'] = filter
    if quality:
        querystring['quality'] = quality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_room_bio(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the bio from a room."
    
    """
    url = f"https://quality-porn.p.rapidapi.com/live-alpha/room-bio"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_tags(gender: str='female', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for live tags and the associated data."
    
    """
    url = f"https://quality-porn.p.rapidapi.com/live-alpha/tags"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_cams(gender: str='female', tag: str='asian', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for live cams and according to your preferences. Filter by gender, tags and pages."
    
    """
    url = f"https://quality-porn.p.rapidapi.com/live-alpha/cams"
    querystring = {}
    if gender:
        querystring['gender'] = gender
    if tag:
        querystring['tag'] = tag
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_search_query(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a random search query;"
    
    """
    url = f"https://quality-porn.p.rapidapi.com/search-queries/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_10_search_queries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top 10 search queries"
    
    """
    url = f"https://quality-porn.p.rapidapi.com/search-queries/top-10"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pornstar_list_by_alphabetical_letter(letter: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Loads a list with the names and request URLs of pornstars. This list is requested by alphabet."
    
    """
    url = f"https://quality-porn.p.rapidapi.com/pornstar/list"
    querystring = {'letter': letter, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quality-porn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

