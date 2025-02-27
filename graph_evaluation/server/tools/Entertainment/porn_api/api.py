import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_tags(query: str='sex', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This ```/tags/search``` endpoint allows users to retrieve a list of tags based on a specific query. Users can input a keyword or phrase, and the endpoint will return a list of relevant tags that can be used to categorize or label content."
    
    """
    url = f"https://porn-api5.p.rapidapi.com/tags/search"
    querystring = {}
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The```tags``` endpoint  returns a list of all the tags associated with the videos available. This endpoint can be used to retrieve metadata about the videos, such as their categories, topics, and keywords, which can be used to improve search results and recommendations for users. The response from this endpoint includes a array of tags."
    
    """
    url = f"https://porn-api5.p.rapidapi.com/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_videos(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The```/videos/search``` endpoint  allows you to search for videos based on specific keywords.  This endpoint can be used to provide users with relevant search results based on their search queries, and can be integrated into search functionality within an application or website."
    
    """
    url = f"https://porn-api5.p.rapidapi.com/videos/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video(title_id: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The ```/video``` endpoint allows developers to retrieve a specific video by passing in its unique ```id``  and ```title_id``` as search parameters. This endpoint returns a video, ready to strean in mp4 format. This endpoint is useful for developers who need to display a specific video on their platform."
    
    """
    url = f"https://porn-api5.p.rapidapi.com/videos/play"
    querystring = {'title_id': title_id, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This ```/videos```  endpoint returns a curated list of the most popular and trending videos ready to stream from various sources. These videos are updated regularly and are sorted based on their popularity and engagement metrics. The endpoint provides a simple and efficient way for developers to access and display the hottest porn videos on their platforms."
    
    """
    url = f"https://porn-api5.p.rapidapi.com/videos"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

