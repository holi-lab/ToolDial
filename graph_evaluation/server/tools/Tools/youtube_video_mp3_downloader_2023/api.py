import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_mp3_of_youtube_video(url: str='https://www.youtube.com/watch?v=M2ErSQhfrcE', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows developers to fetch the details of a YouTube video by providing a valid YouTube video URL. The API extracts relevant metadata from the video including title, thumbnail, channel, and then* gets the Mp3 of the video* and returns the data in a standardized JSON format. This data can be used by developers to display video information in their applications or to enhance search functionality."
    
    """
    url = f"https://youtube-video-mp3-downloader-2023.p.rapidapi.com/api/fetch-video"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-video-mp3-downloader-2023.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

