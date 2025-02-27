import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getappsbydeveloperid(developerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of applications of a single developer by its Id"
    developerid: Developer ID
        
    """
    url = f"https://app-store12.p.rapidapi.com/developers/{developerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsimilarappsbybundleid(bundleid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of similar applications by another bundle Id."
    bundleid: Bundle ID
        
    """
    url = f"https://app-store12.p.rapidapi.com/bundles/{bundleid}/similar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreviewsbybundleid(bundleid: str, page: int=1, country: str='us', sort: str='recent', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of reviews of a single application by its bundle ID"
    bundleid: Bundle ID
        page: Page of the search results
        country: App Store country
        sort: Sort reviews by specific property.
        
    """
    url = f"https://app-store12.p.rapidapi.com/bundles/{bundleid}/reviews"
    querystring = {}
    if page:
        querystring['page'] = page
    if country:
        querystring['country'] = country
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchappsbyterm(term: str, page: int=1, country: str='us', lang: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of apps by term, country and language."
    term: Term that you want to search on App Store.
        page: Page of the search results
        country: App Store country
        lang: Language of the results. If not used, will country's language selected as default.
        
    """
    url = f"https://app-store12.p.rapidapi.com/apps"
    querystring = {'term': term, }
    if page:
        querystring['page'] = page
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreviewsbyappid(appid: str, page: int=1, country: str='us', sort: str='recent', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of reviews of a single application by its application ID"
    appid: Application ID
        page: Page of the search results
        country: App Store country
        sort: Sort reviews by specific property.
        
    """
    url = f"https://app-store12.p.rapidapi.com/apps/{appid}/reviews"
    querystring = {}
    if page:
        querystring['page'] = page
    if country:
        querystring['country'] = country
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getappbyid(appid: str, lang: str='en-us', country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single application by its ID"
    appid: Application ID
        lang: Language of the results. If not used, will country's language selected as default.
        country: App Store country
        
    """
    url = f"https://app-store12.p.rapidapi.com/apps/{appid}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsimilarappsbyappid(appid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of similar applications by another application Id."
    appid: Application ID
        
    """
    url = f"https://app-store12.p.rapidapi.com/apps/{appid}/similar"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getappbybundleid(bundleid: str, lang: str='en-us', country: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about an application by its bundle Id"
    bundleid: Bundle ID
        lang: Language of the results. If not used, will country's language selected as default.
        country: App Store country
        
    """
    url = f"https://app-store12.p.rapidapi.com/bundles/{bundleid}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchappsbytermbundle(term: str, country: str='us', lang: str='en-us', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of apps by term, country and language."
    term: Term that you want to search on App Store.
        country: App Store country
        lang: Language of the results. If not used, will country's language selected as default.
        page: Page of the search results
        
    """
    url = f"https://app-store12.p.rapidapi.com/bundles"
    querystring = {'term': term, }
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "app-store12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

