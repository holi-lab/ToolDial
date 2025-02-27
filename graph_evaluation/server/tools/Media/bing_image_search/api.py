import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_insights(insightstoken: str, query: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns insights about an image, such as webpages that include the image."
    insightstoken: The token from a previous Image API call
        query: The user's search query string
        
    """
    url = f"https://bing-image-search1.p.rapidapi.com/images/details"
    querystring = {'insightsToken': insightstoken, }
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-image-search1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_search(q: str, count: int=None, safesearch: str=None, offset: str=None, mkt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relevant images for a given query."
    q: The user's search query string
        count: The number of image results to return in the response. The actual number delivered may be less than requested.
        safesearch: A filter used to filter results for adult content.
        offset: The zero-based offset that indicates the number of image results to skip before returning results.
        mkt: The market where the results come from. Typically, this is the country where the user is making the request from; however, it could be a different country if the user is not located in a country where Bing delivers results. The market must be in the form -. For example, en-US.



Full list of supported markets:
es-AR, en-AU, de-AT, nl-BE, fr-BE, pt-BR, en-CA, fr-CA, es-CL, da-DK, fi-FI, fr-FR, de-DE, zh-HK, en-IN, en-ID, en-IE, it-IT, ja-JP, ko-KR, en-MY, es-MX, nl-NL, en-NZ, no-NO, zh-CN, pl-PL, pt-PT, en-PH, ru-RU, ar-SA, en-ZA, es-ES, sv-SE, fr-CH, de-CH, zh-TW, tr-TR, en-GB, en-US, es-US
        
    """
    url = f"https://bing-image-search1.p.rapidapi.com/images/search"
    querystring = {'q': q, }
    if count:
        querystring['count'] = count
    if safesearch:
        querystring['safeSearch'] = safesearch
    if offset:
        querystring['offset'] = offset
    if mkt:
        querystring['mkt'] = mkt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-image-search1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currently trending images."
    
    """
    url = f"https://bing-image-search1.p.rapidapi.com/images/trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-image-search1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

