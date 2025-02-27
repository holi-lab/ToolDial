import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def canvas_titles_get_info(titleno: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comic information
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    titleno: The value of titleNo field returned in .../canvas/titles/list or .../canvas/search or .../canvas/home endpoint
        language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/titles/get-info"
    querystring = {'titleNo': titleno, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canvas_titles_get_recommend(titleno: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similar comics
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    titleno: The value of titleNo field returned in .../canvas/titles/list or .../canvas/search or .../canvas/home endpoint
        language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/titles/get-recommend"
    querystring = {'titleNo': titleno, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canvas_search(query: str, language: str='en', pagesize: int=20, startindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for comics by term or phrase
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    language: One of the following : en|zh-hant|de|fr|es|th|id
        pagesize: For paging purpose. Maximum is 20.
        startindex: For paging purpose
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/search"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    if pagesize:
        querystring['pageSize'] = pagesize
    if startindex:
        querystring['startIndex'] = startindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canvas_genres_list(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List genres in canvas category
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/genres/list"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canvas_titles_list(genre: str, sortorder: str='READ_COUNT', language: str='en', startindex: int=0, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List comics in canvas category
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    genre: The value of code field returned in .../canvas/genres/list endpoint.
        sortorder: One of the following : UPDATE|READ_COUNT|LIKEIT
        language: One of the following : en|zh-hant|de|fr|es|th|id
        startindex: For paging purpose
        pagesize: For paging purpose. Maximum is 20.
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/titles/list"
    querystring = {'genre': genre, }
    if sortorder:
        querystring['sortOrder'] = sortorder
    if language:
        querystring['language'] = language
    if startindex:
        querystring['startIndex'] = startindex
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def originals_titles_get_info(titleno: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comic information
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    titleno: The value of titleNo field returned in .../originals/titles/list or .../originals/titles/list-by-rank
        language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/originals/titles/get-info"
    querystring = {'titleNo': titleno, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def originals_episodes_get_likes(titleno: int, episodeno: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get likes count of a episode"
    titleno: The value of titleNo field returned in .../originals/titles/list or .../originals/titles/list-by-rank endpoint
        episodeno: The value of episodeNo field returned in .../originals/episodes/list endpoint.
        language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/originals/episodes/get-likes"
    querystring = {'titleNo': titleno, 'episodeNo': episodeno, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def originals_titles_get_recommend(titleno: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similar comics
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    titleno: The value of titleNo field returned in .../originals/titles/list or .../originals/titles/list-by-rank endpoint
        language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/originals/titles/get-recommend"
    querystring = {'titleNo': titleno, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def originals_episodes_get_info(episodeno: int, titleno: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get episode and photo links
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    episodeno: The value of episodeNo field returned in .../originals/episodes/list endpoint.
        titleno: The value of titleNo field returned in .../originals/titles/list or .../originals/titles/list-by-rank endpoint
        language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/originals/episodes/get-info"
    querystring = {'episodeNo': episodeno, 'titleNo': titleno, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def originals_episodes_list(titleno: int, language: str='en', pagesize: int=20, startindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List episodes of a comic
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    titleno: The value of titleNo field returned in .../originals/titles/list or .../originals/titles/list-by-rank endpoint
        language: One of the following : en|zh-hant|de|fr|es|th|id
        pagesize: For paging purpose. Maximum is 20.
        startindex: For paging purpose
        
    """
    url = f"https://webtoon.p.rapidapi.com/originals/episodes/list"
    querystring = {'titleNo': titleno, }
    if language:
        querystring['language'] = language
    if pagesize:
        querystring['pageSize'] = pagesize
    if startindex:
        querystring['startIndex'] = startindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canvas_episodes_get_info(episodeno: int, titleno: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get episode and photo links.
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    episodeno: The value of episodeNo field returned in .../canvas/episodes/list endpoint.
        titleno: The value of titleNo field returned in .../canvas/titles/list or .../canvas/search endpoint
        language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/episodes/get-info"
    querystring = {'episodeNo': episodeno, 'titleNo': titleno, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canvas_episodes_list(titleno: int, language: str='en', pagesize: int=20, startindex: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List episodes of a comic
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    titleno: The value of titleNo field returned in .../canvas/titles/list or .../canvas/search or .../canvas/home endpoint
        language: One of the following : en|zh-hant|de|fr|es|th|id
        pagesize: For paging purpose. Maximum is 20.
        startindex: For paging purpose
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/episodes/list"
    querystring = {'titleNo': titleno, }
    if language:
        querystring['language'] = language
    if pagesize:
        querystring['pageSize'] = pagesize
    if startindex:
        querystring['startIndex'] = startindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def originals_genres_list(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List genres in originals category"
    language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/originals/genres/list"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canvas_episodes_get_likes(episodeno: int, titleno: int, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get likes count of a episode"
    episodeno: The value of episodeNo field returned in .../canvas/episodes/list endpoint.
        titleno: The value of titleNo field returned in .../canvas/titles/list or .../canvas/search endpoint
        language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/episodes/get-likes"
    querystring = {'episodeNo': episodeno, 'titleNo': titleno, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canvas_home(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reproduce comic data in home screen
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/canvas/home"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def originals_titles_list(language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List comics in originals category
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    language: One of the following : en|zh-hant|de|fr|es|th|id
        
    """
    url = f"https://webtoon.p.rapidapi.com/originals/titles/list"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def originals_titles_list_by_rank(language: str='en', count: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List comics in originals category by ranking
		*To load images, please check the tutorial at https://rapidapi.com/apidojo/api/webtoon/tutorials/how-to-load-images"
    language: One of the following : en|zh-hant|de|fr|es|th|id
        count: The number of comics with highest ranked per genre. Maximum is 30.
        
    """
    url = f"https://webtoon.p.rapidapi.com/originals/titles/list-by-rank"
    querystring = {}
    if language:
        querystring['language'] = language
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webtoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

