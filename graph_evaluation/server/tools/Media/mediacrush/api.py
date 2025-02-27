import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_hash_delete(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns: a dictionary describing whether the delete operation succeeded. In most cases it is easier to check the HTTP status code."
    
    """
    url = f"https://community-mediacrush.p.rapidapi.com/{hash}/delete"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-mediacrush.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_hash_status(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns: the processing status of the file identified by <hash>."
    
    """
    url = f"https://community-mediacrush.p.rapidapi.com/{hash}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-mediacrush.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_hash_exists(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns: a dictionary answering the question of whether a hash exists."
    
    """
    url = f"https://community-mediacrush.p.rapidapi.com/{hash}/exists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-mediacrush.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_upload_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameters: url, the URL from where to fetch the file to upload.  Returns: the same as /api/upload/file."
    url: the URL from where to fetch the file to upload.
        
    """
    url = f"https://community-mediacrush.p.rapidapi.com/upload/url"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-mediacrush.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_hash(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns: information about the file whose hash is <hash>. When a file is uploaded to MediaCrush, several associated files may be generated. In the case of GIF files, two video files are generated - one with h.264/mpeg and another with theora/vorbis. Some media will also have "extra" files. In the case of uploaded videos, we'll include an image/png thumbnail file in the extras."
    
    """
    url = f"https://community-mediacrush.p.rapidapi.com/{hash}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-mediacrush.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_info(list: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns: an array of file objects."
    list: a comma-separated list of hashes.
        
    """
    url = f"https://community-mediacrush.p.rapidapi.com/info"
    querystring = {'list': list, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-mediacrush.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

