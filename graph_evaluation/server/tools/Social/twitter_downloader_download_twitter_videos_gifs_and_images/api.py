import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_private_tweet_data(url: str, cookie: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all information about a specific tweet on Twitter."
    cookie: Please log in to your account and obtain a valid cookie. To access the requested resource, you need to include a valid cookie in your API request. 
        
    """
    url = f"https://twitter-downloader-download-twitter-videos-gifs-and-images.p.rapidapi.com/tweet"
    querystring = {'url': url, 'Cookie': cookie, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-downloader-download-twitter-videos-gifs-and-images.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tweet_data(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return back all information about a specific tweet on Twitter."
    
    """
    url = f"https://twitter-downloader-download-twitter-videos-gifs-and-images.p.rapidapi.com/status"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-downloader-download-twitter-videos-gifs-and-images.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

