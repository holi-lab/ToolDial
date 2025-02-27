import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def livetv(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of the live streaming information for news channels in the specified language. Use this endpoint to get the live streaming ID, channel name, and channel logo link for all news channels available in the specified language.
		
		**Response Format**
		The API response will be in JSON format and will include an array of objects, where each object represents a news channel and includes the following properties:
		
		id: A unique identifier for the live streaming of the news channel.
		name: The name of the news channel.
		link: The URL for the live streaming of the news channel.
		logo: The URL of the logo for the news channel."
    
    """
    url = f"https://live-news-channel-api-access-youtube-streams-in-27-languages.p.rapidapi.com/livetv/{language}/.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-news-channel-api-access-youtube-streams-in-27-languages.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns a list of the available languages for the Live News Channel API. Use this endpoint to get a list of all the languages supported by the API before making requests for live news channels in a specific language."
    
    """
    url = f"https://live-news-channel-api-access-youtube-streams-in-27-languages.p.rapidapi.com/languages/.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-news-channel-api-access-youtube-streams-in-27-languages.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

