import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v4_contents_search(intent: str, gender: str, page: int=0, countrycode: str=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use Bobble AI’s powerful recommendation engine to get content from our vast library consisting of thousands of tagged domain and intent-specific animated stickers and movie GIFs"
    intent: The intent against which the content has to be returned.
        gender: The gender on which the content is based. Can be male, female or unisex
        page: The page number of the content to be fetched.
        countrycode: 2 letter ISO country code of the country for which the country has to be filtered against. If the API is integrated on the server side then this param can be used, if it is being integrated client side, then the IP address of the client will be used to detect the country code (if required). Send as empty to avoid filtering.
        limit: The maximum number of results to return
        
    """
    url = f"https://ai-powered-contextual-rich-media-content-recommendation.p.rapidapi.com/v4/contents/search"
    querystring = {'intent': intent, 'gender': gender, }
    if page:
        querystring['page'] = page
    if countrycode:
        querystring['countryCode'] = countrycode
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-powered-contextual-rich-media-content-recommendation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

