import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_files(apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the files that are related to your storage container."
    apikey: The API key related to your account. You can get this by making a request to the  /generate-api-key endpoint
        
    """
    url = f"https://veestream2.p.rapidapi.com/files"
    querystring = {'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veestream2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_video_gif(is_id: str, apikey: str='77041892a9750c4041c1f0d2e24f6f459d8a71d2', start: str='10', end: str='12', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generates gif image from a video file. You can pass start and end in the query to specify where you want your gif to start and end."
    id: The unique id for the video. After uploading a file you get _id as part of the response. Use _id as your id
        apikey: The apikey you got after calling the /generate-api-key endpoint
        start: Where you want your gif to start (secs)
        end: Where you want your gif to end (secs)
        
    """
    url = f"https://veestream2.p.rapidapi.com/file/{is_id}/gif"
    querystring = {}
    if apikey:
        querystring['apikey'] = apikey
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veestream2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_thumbnail(apikey: str, is_id: str, start: int=3, end: int=4, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generates a thumbnail from a video file. It will only work for video files. The start and end values in the query lets you specify where you want to screenshot your video."
    id: The id parameter is the unique identifier for your file. It is the _id from the upload response.
        start: Where you want your the image to be shot from. E.g 10  for 10 secs.
        end: The section of the video to take the shot from will be from start to end where start is what you provide in the start query and end is what you put here.
        
    """
    url = f"https://veestream2.p.rapidapi.com/file/{is_id}/poster"
    querystring = {'apikey': apikey, }
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veestream2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_file_metadata(is_id: str, apikey: str='77041892a9750c4041c1f0d2e24f6f459d8a71d2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns metadata about a particular file. The metadata includes the name, _id, url, gif and poster. The latter 2 are useful for videos since they are endpoints to generate gifs and image thumbnails from videos. For non video files, this endpoints will be not useful. We have intentionally decided to let them be in the response for non video files though they do nothing."
    
    """
    url = f"https://veestream2.p.rapidapi.com/file/{is_id}/data"
    querystring = {}
    if apikey:
        querystring['apikey'] = apikey
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veestream2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_file_progressive_download_url(apikey: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A GET request that allows you to download a file from your storage container that has your files. The file can be an image, video, or any other file type. In order to access the endpoint, you need to pass an "apikey" header in the request headers.
		
		To use this endpoint, replace "{id}" with the "_id" property of the file you want to download. This will return a valid file URL that you can use to download the file.
		
		Overall, this endpoint provides a simple and secure way to download files from your container with the added benefit of being able to work with a variety of file types."
    
    """
    url = f"https://veestream2.p.rapidapi.com/file/{is_id}"
    querystring = {'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veestream2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

