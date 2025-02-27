import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def imagesearch(q: str, pagenumber: int, pagesize: int, autocorrect: bool, safesearch: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relevant images for a given query."
    q: The user's search query string.
        pagenumber: The page to view.
        pagesize: The number of items per page. The maximum value is 50.
        autocorrect: Automatically correct spelling.
        safesearch: A filter used to filter results for adult content.
        
    """
    url = f"https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/ImageSearchAPI"
    querystring = {'q': q, 'pageNumber': pagenumber, 'pageSize': pagesize, 'autoCorrect': autocorrect, }
    if safesearch:
        querystring['safeSearch'] = safesearch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contextualwebsearch-websearch-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def newssearch(pagesize: int, autocorrect: bool, q: str, pagenumber: int, topublisheddate: str='null', safesearch: bool=None, frompublisheddate: str='null', withthumbnails: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news articles relevant for a given query."
    pagesize: The number of items per page. The maximum value is 50.
        autocorrect: Automatically correct spelling.
        q: The user's search query string.
        pagenumber: The page to view.
        topublisheddate: The  published date and time for the newest article allowed.  For example: *2015-05-16T05:50:06.* See  [https://www.c-sharpcorner.com/blogs/date-and-time-format-in-c-sharp-programming1 ](url)for more possible DateTime formats. 
        safesearch: A filter used to filter results for adult content.
        frompublisheddate: The  published date and time for the oldest article allowed.  For example: *2015-05-16T05:50:06.* See  [https://www.c-sharpcorner.com/blogs/date-and-time-format-in-c-sharp-programming1 ](url)for more possible DateTime formats. 
        withthumbnails: Show results with image thumbnails.
        
    """
    url = f"https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"
    querystring = {'pageSize': pagesize, 'autoCorrect': autocorrect, 'q': q, 'pageNumber': pagenumber, }
    if topublisheddate:
        querystring['toPublishedDate'] = topublisheddate
    if safesearch:
        querystring['safeSearch'] = safesearch
    if frompublisheddate:
        querystring['fromPublishedDate'] = frompublisheddate
    if withthumbnails:
        querystring['withThumbnails'] = withthumbnails
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contextualwebsearch-websearch-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def websearch(q: str, autocorrect: bool, pagenumber: int, pagesize: int, safesearch: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relevant web pages for a given query."
    q: The user's search query string.
        autocorrect: Automatically correct spelling.
        pagenumber: The page to view.
        pagesize: The number of items per page. The maximum value is 50.
        safesearch: A filter used to filter results for adult content.
        
    """
    url = f"https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/WebSearchAPI"
    querystring = {'q': q, 'autoCorrect': autocorrect, 'pageNumber': pagenumber, 'pageSize': pagesize, }
    if safesearch:
        querystring['safeSearch'] = safesearch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contextualwebsearch-websearch-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spellcheck(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check spelling."
    text: The text string to check for spelling errors.
        
    """
    url = f"https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/spelling/SpellCheck"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contextualwebsearch-websearch-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Suggest as-you-type completion."
    text: The prefix to complete
        
    """
    url = f"https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/spelling/AutoComplete"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "contextualwebsearch-websearch-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

