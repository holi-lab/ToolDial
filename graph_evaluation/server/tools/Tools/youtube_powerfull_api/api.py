import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_video_keywords(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By Providing The Video Link You can GET the Video Keywords"
    
    """
    url = f"https://youtube-powerfull-api.p.rapidapi.com/api/get_video_kewords/{video_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-powerfull-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_html_file_provide_the_playlist_id(playlist_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By giving the target playlist ID yon can get that specific playlist HTML generated file."
    
    """
    url = f"https://youtube-powerfull-api.p.rapidapi.com/api/get_playlist_html_file/{playlist_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-powerfull-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_full_details_provide_playlist_id(playlist_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Huge data of a given Playlist ID."
    
    """
    url = f"https://youtube-powerfull-api.p.rapidapi.com/api/get_playlist_full_details/{playlist_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-powerfull-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_playlist_basic_details_provide_youtube_playlist_id(playlist_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Basic Information of a playlist by providing the playlist ID."
    
    """
    url = f"https://youtube-powerfull-api.p.rapidapi.com/api/get_playlist_details/{playlist_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-powerfull-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_s_full_details_provide_the_video_id(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve full information about a video by simply providing the video ID.  
		If you don't know where to find the video ID ? kindly check the bellow example.
		https://www.youtube.com/watch?v=**nu_pCVPKzTk**
		As you see in the above link a string inside the double stars is the video ID."
    
    """
    url = f"https://youtube-powerfull-api.p.rapidapi.com/api/get_full_video_details/{video_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-powerfull-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_s_basic_information_provide_video_id(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By simply providing the video ID, Get useful and complete information about the target video."
    
    """
    url = f"https://youtube-powerfull-api.p.rapidapi.com/api/get_video_info/{video_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-powerfull-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_single_video_provide_video_id(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By Providing the video ID simply download any video from YouTube."
    
    """
    url = f"https://youtube-powerfull-api.p.rapidapi.com/download/single_video/{video_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-powerfull-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

