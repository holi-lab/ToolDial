import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def customfeedapiv2(searchengineid: str, pagenumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news articles relevant to your custom feed."
    searchengineid: The id of the custom search engine. To setup a custom search engine go to: [cse.usearch.com](url)
        pagenumber: The page to view.
        
    """
    url = f"https://custom-search.p.rapidapi.com/api/search/CustomFeedAPIV2"
    querystring = {'searchEngineId': searchengineid, 'pageNumber': pagenumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "custom-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def customimagesearchapiv2(pagenumber: int, searchengineid: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relevant images for a given query."
    pagenumber: The page to view.
        searchengineid: The id of the custom search engine. To setup a custom search engine go to: [cse.usearch.com](url)
        q: The user's search query string.
        
    """
    url = f"https://custom-search.p.rapidapi.com/api/search/CustomImageSearchAPIV2"
    querystring = {'pageNumber': pagenumber, 'searchEngineId': searchengineid, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "custom-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def customnewssearchapiv2(q: str, pagenumber: int, searchengineid: str, frompublisheddate: str=None, topublisheddate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news articles relevant for a given query."
    q: The user's search query string.
        pagenumber: The page to view.
        searchengineid: The id of the custom search engine. To setup a custom search engine go to: [cse.usearch.com](url)
        frompublisheddate: The  published date and time for the oldest article allowed.  For example: *2015-05-16T05:50:06.* See  [https://www.c-sharpcorner.com/blogs/date-and-time-format-in-c-sharp-programming1 ](url)for more possible DateTime formats. 
        topublisheddate: The  published date and time for the newest article allowed.  For example: *2015-05-16T05:50:06.* See  [https://www.c-sharpcorner.com/blogs/date-and-time-format-in-c-sharp-programming1 ](url)for more possible DateTime formats. 
        
    """
    url = f"https://custom-search.p.rapidapi.com/api/search/CustomNewsSearchAPIV2"
    querystring = {'q': q, 'pageNumber': pagenumber, 'searchEngineId': searchengineid, }
    if frompublisheddate:
        querystring['fromPublishedDate'] = frompublisheddate
    if topublisheddate:
        querystring['toPublishedDate'] = topublisheddate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "custom-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def customwebsearchapiv2(pagenumber: int, searchengineid: str, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relevant web pages for a given query."
    pagenumber: The page to view.
        searchengineid: The id of the custom search engine. To setup a custom search engine go to: [cse.usearch.com](url)
        q: The user's search query string.
        
    """
    url = f"https://custom-search.p.rapidapi.com/api/search/CustomWebSearchAPIV2"
    querystring = {'pageNumber': pagenumber, 'searchEngineId': searchengineid, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "custom-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def customnewssearchapiv3(pagenumber: int, pagesize: int, q: str, searchengineid: str, topublisheddate: str=None, frompublisheddate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news articles relevant for a given query."
    pagenumber: The page to view.
        pagesize: The page size.
        q: The user's search query string.
        searchengineid: The id of the custom search engine. To setup a custom search engine go to: [cse.usearch.com](url)
        topublisheddate: The  published date and time for the newest article allowed.  For example: *2015-05-16T05:50:06.* See  [https://www.c-sharpcorner.com/blogs/date-and-time-format-in-c-sharp-programming1 ](url)for more possible DateTime formats. 
        frompublisheddate: The  published date and time for the oldest article allowed.  For example: *2015-05-16T05:50:06.* See  [https://www.c-sharpcorner.com/blogs/date-and-time-format-in-c-sharp-programming1 ](url)for more possible DateTime formats. 
        
    """
    url = f"https://custom-search.p.rapidapi.com/api/search/CustomNewsSearchAPIV3"
    querystring = {'pageNumber': pagenumber, 'pageSize': pagesize, 'q': q, 'searchEngineId': searchengineid, }
    if topublisheddate:
        querystring['toPublishedDate'] = topublisheddate
    if frompublisheddate:
        querystring['fromPublishedDate'] = frompublisheddate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "custom-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

