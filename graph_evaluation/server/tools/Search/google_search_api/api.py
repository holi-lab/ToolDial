import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(keyword: str, num: int=None, start: int=None, html: str=None, uule: str=None, language: str=None, device: str='Desktop', country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search results for Google search keyword query. Customize your results using various parameters"
    num: Number of results. If empty, defaults to 10
        start: Defines how many results to skip. You can enter start= in your google URL to check this. For 2nd page of results, you would use start=10. If empty, defaults to 0
        html: You can choose to get the full page HTML of google page in your result
        uule: Google's UULE param allows you to specify where the results should be gotten. You have to calculate the UULE yourself and pass it here. We will be adding a location param soon where you can just enter the name of the location and we will calculate UULE ourselves (New York, NY, USA)
        language: Full language name. E.g. English, Italian, Hindi, Spanish, etc
        device: Either 'desktop' or 'mobile'. Google displays different results based on device so choose the one you care about. If empty, defaults to 'desktop'
        country: Full name of the country (e.g. United States, Canada, Germany, etc.). This will return results based on that specific country's google tld (.ca, .co.uk, .de).
        
    """
    url = f"https://google-search-api7.p.rapidapi.com/search"
    querystring = {'keyword': keyword, }
    if num:
        querystring['num'] = num
    if start:
        querystring['start'] = start
    if html:
        querystring['html'] = html
    if uule:
        querystring['uule'] = uule
    if language:
        querystring['language'] = language
    if device:
        querystring['device'] = device
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-api7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

