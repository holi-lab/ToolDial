import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def photos_by_actor_with_count(service: str, count: int, actor: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get multiple images of a specific performer from a large library.
		
		A list: https://api.ofhub.fun/actors"
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Service': service, 'Count': count, 'Actor': actor, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_photos_by_count(count: int, service: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get multiple images from a large gallery."
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Count': count, 'Service': service, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos_by_duration_with_actor_and_count(minduration: str, service: str, actor: str, count: int, maxduration: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve multiple videos with specific timestamps and a specific performer from a large library.
		
		A list: https://api.ofhub.fun/actors"
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'MINDURATION': minduration, 'Service': service, 'Actor': actor, 'Count': count, 'MAXDURATION': maxduration, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos_by_duration_with_count(minduration: str, maxduration: str, service: str, count: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve multiple videos with specific timestamps from a large library."
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'MINDURATION': minduration, 'MAXDURATION': maxduration, 'Service': service, 'Count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_by_duration_with_actor(maxduration: str, minduration: str, service: str, actor: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a video with specific timestamps from a specific performer from a large library.
		
		A list: https://api.ofhub.fun/actors"
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'MAXDURATION': maxduration, 'MINDURATION': minduration, 'Service': service, 'Actor': actor, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos_by_actor_with_count(service: str, actor: str, count: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get multiple videos of a specific performer from a large library.
		
		A list: https://api.ofhub.fun/actors"
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Service': service, 'Actor': actor, 'Count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_by_duration(service: str, minduration: str, maxduration: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a video with specific timestamps from a large library."
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Service': service, 'MINDURATION': minduration, 'MAXDURATION': maxduration, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_videos_by_count(service: str, count: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get multiple videos from a large gallery."
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Service': service, 'Count': count, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_by_actor(actor: str, service: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a video of a specific performer from a large gallery.
		
		A list: https://api.ofhub.fun/actors"
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Actor': actor, 'Service': service, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def photo_by_actor(service: str, actor: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a picture of a specific performer from a large gallery.
		
		A list: https://api.ofhub.fun/actors"
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Service': service, 'Actor': actor, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_photo(service: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a picture from a large gallery."
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Service': service, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_video(service: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a video from a large gallery."
    
    """
    url = f"https://porn-as-a-service.p.rapidapi.com/"
    querystring = {'Service': service, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "porn-as-a-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

