import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trending(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currently trending videos."
    
    """
    url = f"https://bing-video-search1.p.rapidapi.com/videos/trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-video-search1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_details(modules: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get insights about a video, such as related videos."
    
    """
    url = f"https://bing-video-search1.p.rapidapi.com/videos/details"
    querystring = {'modules': modules, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-video-search1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_search(q: str, safesearch: str=None, mkt: str=None, count: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos relevant for a given query."
    q: The user's search query string
        safesearch: A filter used to filter results for adult content.
        mkt: The market where the results come from. Typically, this is the country where the user is making the request from; however, it could be a different country if the user is not located in a country where Bing delivers results. The market must be in the form -. For example, en-US.

Full list of supported markets:
es-AR, en-AU, de-AT, nl-BE, fr-BE, pt-BR, en-CA, fr-CA, es-CL, da-DK, fi-FI, fr-FR, de-DE, zh-HK, en-IN, en-ID, en-IE, it-IT, ja-JP, ko-KR, en-MY, es-MX, nl-NL, en-NZ, no-NO, zh-CN, pl-PL, pt-PT, en-PH, ru-RU, ar-SA, en-ZA, es-ES, sv-SE, fr-CH, de-CH, zh-TW, tr-TR, en-GB, en-US, es-US
        count: The number of video results to return in the response. The actual number delivered may be less than requested.
        offset: The zero-based offset that indicates the number of video results to skip before returning results.
        
    """
    url = f"https://bing-video-search1.p.rapidapi.com/videos/search"
    querystring = {'q': q, }
    if safesearch:
        querystring['safeSearch'] = safesearch
    if mkt:
        querystring['mkt'] = mkt
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-video-search1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

