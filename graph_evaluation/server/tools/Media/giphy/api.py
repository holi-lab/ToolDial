import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, limit: str=None, offset: str=None, rating: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search all Giphy GIFs for a word or phrase. Punctuation will be stripped and ignored. Use a plus or url encode for phrases."
    q: search query term or phrase
        limit: number of results to return, maximum 100. Default 25.
        offset: results offset, defaults to 0.
        rating: limit results to those rated (y,g, pg, pg-13 or r)
        
    """
    url = f"https://giphy.p.rapidapi.com/v1/gifs/search"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_gif_by_id(gif_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns meta data about a GIF, by GIF id. In the below example, the GIF ID is "feqkVgjJpYtjy""
    
    """
    url = f"https://giphy.p.rapidapi.com/v1/gifs/{gif_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_gifs_by_id(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A multiget version of the get GIF by ID endpoint. In this case the IDs are feqkVgjJpYtjy and 7rzbxdu0ZEXLy"
    
    """
    url = f"https://giphy.p.rapidapi.com/v1/gifs"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def translate(s: str, rating: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is prototype endpoint for using Giphy as a translation engine for a GIF dialect. The translate API draws on search, but uses the Giphy "special sauce" to handle translating from one vocabulary to another. In this case, words and phrases to GIFs. Use a plus or url encode for phrases."
    s: term or phrase to translate into a GIF
        rating: limit results to those rated (y,g, pg, pg-13 or r).
        
    """
    url = f"https://giphy.p.rapidapi.com/v1/gifs/translate"
    querystring = {'s': s, }
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random(tag: str, rating: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random GIF, limited by tag. Excluding the tag parameter will return a random GIF from the Giphy catalog."
    tag: the GIF tag to limit randomness by
        rating: limit results to those rated (y,g, pg, pg-13 or r).
        
    """
    url = f"https://giphy.p.rapidapi.com/v1/gifs/random"
    querystring = {'tag': tag, }
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_gifs(limit: str=None, rating: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch GIFs currently trending online. The data returned mirrors that used to create The Hot 100 list of GIFs on Giphy. Returns 25 results by default."
    limit: limits the number of results returned. By default returns 25 results.
        rating: limit results to those rated (y,g, pg, pg-13 or r).
        
    """
    url = f"https://giphy.p.rapidapi.com/v1/gifs/trending"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sticker_search(q: str, limit: str=None, offset: str=None, rating: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Replicates the functionality and requirements of the classic Giphy search, but returns animated stickers rather than gifs."
    q: search query term or phrase
        limit: number of results to return, maximum 100. Default 25
        offset: results offset, defaults to 0
        rating: limit results to those rated (y,g, pg, pg-13 or r).
        
    """
    url = f"https://giphy.p.rapidapi.com/v1/stickers/search"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sticker_roulette(tag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a spotaneously selected sticker from Giphy's sticker collection. Optionally limit scope of result to a specific tag. Like the GIF random endpoint, Punctuation will be stripped and ignored. Use a hyphen for phrases. Example oops, birthday or thank-you. Search terms should be URL encoded."
    
    """
    url = f"https://giphy.p.rapidapi.com/v1/stickers/random"
    querystring = {'tag': tag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sticker_trending(s: str, limit: str=None, offset: str=None, fmt: str=None, rating: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest stickers trending on Giphy with this endpoint."
    s: term or phrase to translate into a GIF
        limit: number of results to return, maximum 100. Default: 25
        offset: results offset, defaults to 0
        fmt: return results in html or json format.
        rating: limit results to those rated (y,g, pg, pg-13 or r).
        
    """
    url = f"https://giphy.p.rapidapi.com/v1/stickers/trending"
    querystring = {'s': s, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if fmt:
        querystring['fmt'] = fmt
    if rating:
        querystring['rating'] = rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sticker_translate(s: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the same alogirithm as the GIF translate endpoint, the sticker translate endpoint turns words into stickers."
    s: term or phrase to translate into a gif
        
    """
    url = f"https://giphy.p.rapidapi.com/v1/stickers/translate"
    querystring = {'s': s, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "giphy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

