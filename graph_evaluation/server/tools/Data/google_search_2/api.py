import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def google_maps_reviews(data_id: str, next_page_token: str=None, sort_by: str=None, topic_id: str=None, hl: str='en_us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API you can get the JSON data of the review results."
    
    """
    url = f"https://google-search-2.p.rapidapi.com/reviews"
    querystring = {'data_id': data_id, }
    if next_page_token:
        querystring['next_page_token'] = next_page_token
    if sort_by:
        querystring['sort_by'] = sort_by
    if topic_id:
        querystring['topic_id'] = topic_id
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_maps_data_id(query: str, gl: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API you can get the JSON results of the data ID of a particular place."
    
    """
    url = f"https://google-search-2.p.rapidapi.com/dataId"
    querystring = {'query': query, }
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_images(query: str, gl: str='us', hl: str='en_us', duration: str=None, chips: str=None, lr: str=None, ijn: str='0', html: str=None, uule: str=None, device: str='desktop', safe: str=None, nfpr: str=None, tbs: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API you can get the JSON data of the image results."
    
    """
    url = f"https://google-search-2.p.rapidapi.com/images"
    querystring = {'query': query, }
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    if duration:
        querystring['duration'] = duration
    if chips:
        querystring['chips'] = chips
    if lr:
        querystring['lr'] = lr
    if ijn:
        querystring['ijn'] = ijn
    if html:
        querystring['html'] = html
    if uule:
        querystring['uule'] = uule
    if device:
        querystring['device'] = device
    if safe:
        querystring['safe'] = safe
    if nfpr:
        querystring['nfpr'] = nfpr
    if tbs:
        querystring['tbs'] = tbs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_videos(query: str, uule: str=None, duration: str=None, safe: str=None, hl: str='en_us', num: str='10', lr: str=None, device: str='desktop', tbs: str=None, nfpr: str='0', gl: str='us', page: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API you can get the JSON data of the video results."
    
    """
    url = f"https://google-search-2.p.rapidapi.com/videos"
    querystring = {'query': query, }
    if uule:
        querystring['uule'] = uule
    if duration:
        querystring['duration'] = duration
    if safe:
        querystring['safe'] = safe
    if hl:
        querystring['hl'] = hl
    if num:
        querystring['num'] = num
    if lr:
        querystring['lr'] = lr
    if device:
        querystring['device'] = device
    if tbs:
        querystring['tbs'] = tbs
    if nfpr:
        querystring['nfpr'] = nfpr
    if gl:
        querystring['gl'] = gl
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_news(hl: str='en_us', tbs: str=None, lr: str=None, gl: str='us', page: str='0', uule: str=None, html: str=None, duration: str=None, query: str='football', safe: str=None, device: str='desktop', nfpr: str='0', num: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API you can get the JSON data of the news results."
    
    """
    url = f"https://google-search-2.p.rapidapi.com/news"
    querystring = {}
    if hl:
        querystring['hl'] = hl
    if tbs:
        querystring['tbs'] = tbs
    if lr:
        querystring['lr'] = lr
    if gl:
        querystring['gl'] = gl
    if page:
        querystring['page'] = page
    if uule:
        querystring['uule'] = uule
    if html:
        querystring['html'] = html
    if duration:
        querystring['duration'] = duration
    if query:
        querystring['query'] = query
    if safe:
        querystring['safe'] = safe
    if device:
        querystring['device'] = device
    if nfpr:
        querystring['nfpr'] = nfpr
    if num:
        querystring['num'] = num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete_results(query: str, hl: str='en_US', gl: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this API to get suggestions based on the entered keyword."
    
    """
    url = f"https://google-search-2.p.rapidapi.com/autocomplete"
    querystring = {'query': query, }
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def organic_results(query: str, device: str='desktop', hl: str='en', html: str=None, duration: str=None, lr: str=None, safe: str=None, uule: str=None, nfpr: str='0', num: str='10', tbs: str=None, gl: str='us', page: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this API you can get the JSON data of the organic search results."
    
    """
    url = f"https://google-search-2.p.rapidapi.com/organicResults"
    querystring = {'query': query, }
    if device:
        querystring['device'] = device
    if hl:
        querystring['hl'] = hl
    if html:
        querystring['html'] = html
    if duration:
        querystring['duration'] = duration
    if lr:
        querystring['lr'] = lr
    if safe:
        querystring['safe'] = safe
    if uule:
        querystring['uule'] = uule
    if nfpr:
        querystring['nfpr'] = nfpr
    if num:
        querystring['num'] = num
    if tbs:
        querystring['tbs'] = tbs
    if gl:
        querystring['gl'] = gl
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

