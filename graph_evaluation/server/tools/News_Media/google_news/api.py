import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def supported_languages_and_regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to retrieve a list of supported languages and regions."
    
    """
    url = f"https://google-news13.p.rapidapi.com/languageRegions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suggest(keyword: str, lr: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get autocomplete suggestions or query predictions as a user types a search query. The endpoint requires the **keyword** parameter, which represents the partial text entered by the user. You can send a request with the partial text, and the request will generate a JSON response containing a list of relevant autocomplete suggestions for the search query."
    keyword:  The mandatory parameter  to specify the search term
        lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/search/suggest"
    querystring = {'keyword': keyword, }
    if lr:
        querystring['lr'] = lr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(keyword: str, lr: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to search for news from Google News based on keywords. The mandatory parameter to be used is **keyword** to specify the search term"
    keyword:  The mandatory parameter  to specify the search term
        lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/search"
    querystring = {'keyword': keyword, }
    if lr:
        querystring['lr'] = lr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def world(lr: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get world news from Google News. The optional parameter that can be used is "lr" to determine the region"
    
    """
    url = f"https://google-news13.p.rapidapi.com/world"
    querystring = {}
    if lr:
        querystring['lr'] = lr
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport(lr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get sport news from Google News."
    lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/sport"
    querystring = {'lr': lr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def science(lr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get science news from Google News."
    lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/science"
    querystring = {'lr': lr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health(lr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get health  news from Google News."
    lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/health"
    querystring = {'lr': lr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def entertainment(lr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get entertainment news from Google News."
    lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/entertainment"
    querystring = {'lr': lr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business(lr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get business news from Google News."
    lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/business"
    querystring = {'lr': lr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest(lr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get the latest news from Google News."
    lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/latest"
    querystring = {'lr': lr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technology(lr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get technology news from Google News."
    lr: language region, ex: en-US
        
    """
    url = f"https://google-news13.p.rapidapi.com/technology"
    querystring = {'lr': lr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-news13.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

