import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getapppermissionsbyid(appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the permissions of a single application by its ID"
    appid: Application ID
        
    """
    url = f"https://google-play9.p.rapidapi.com/apps/{appid}/permissions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-play9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreviewsbyappid(country: str, appid: str, sort: str='NEWEST', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the reviews of a single application by its ID"
    country: Country for the Google Play
        appid: Application ID
        sort: Country for the Google Play
        lang: Language of the results
        
    """
    url = f"https://google-play9.p.rapidapi.com/apps/{appid}/reviews"
    querystring = {'country': country, }
    if sort:
        querystring['sort'] = sort
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-play9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getappbyid(appid: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single application by its ID"
    appid: Application ID
        lang: Language for the application
        
    """
    url = f"https://google-play9.p.rapidapi.com/apps/{appid}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-play9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsimilarappsbyid(appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of applications that is similar to a single application by its ID"
    appid: Application ID
        
    """
    url = f"https://google-play9.p.rapidapi.com/apps/{appid}/similar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-play9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getapps(search: str, price: str='all', lang: str='en', country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of apps and a pagination token"
    search: Search Term for the Google Play
        price: Price of the applications
        lang: Language of the results
        country: Country for the Google Play
        
    """
    url = f"https://google-play9.p.rapidapi.com/apps"
    querystring = {'search': search, }
    if price:
        querystring['price'] = price
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-play9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getappdatasafetybyid(appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the data safety attributes of a single application by its ID"
    appid: Application ID
        
    """
    url = f"https://google-play9.p.rapidapi.com/apps/{appid}/datasafety"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-play9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getappsbydeveloperid(developerid: str, country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of applications of a single developer by its ID"
    developerid: Developer ID
        country: Country for the Google Play
        
    """
    url = f"https://google-play9.p.rapidapi.com/developers/{developerid}"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-play9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

