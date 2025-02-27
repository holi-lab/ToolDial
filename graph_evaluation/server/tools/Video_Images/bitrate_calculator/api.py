import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def quality_to_1600x900_video_type_1(audio: str, qpp: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 60
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 192"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1600_900/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1600x900_video_type_2(qpp: str, fps: str, audio: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 48
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 160"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1600_900/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1600x900_video_type_3(audio: str, qpp: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 45
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 128"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1600_900/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1600x900_video_type_4(qpp: str, fps: str, audio: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 30
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 96"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1600_900/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1536x864_video_type_3(audio: str, qpp: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 45
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 128"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1536_864/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1280x720_video_type_5(audio: str, fps: str, qpp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 24
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 96"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1280_720/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1920x1080_video_type_5(qpp: str, audio: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 24
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 96"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1920_1080/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1280x720_video_type_4(qpp: str, fps: str, audio: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 30
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 96"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1280_720/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1280x720_video_type_3(qpp: str, audio: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 45
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 128"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1280_720/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1280x720_video_type_2(fps: str, qpp: str, audio: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 48
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 160"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1280_720/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1280x720_video_type_1(audio: str, qpp: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 60
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 192"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1280_720/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1536x864_video_type_5(qpp: str, fps: str, audio: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 24
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 96"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1536_864/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1536x864_video_type_4(fps: str, audio: str, qpp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 30
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 96"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1536_864/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1536x864_video_type_2(qpp: str, fps: str, audio: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 48
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 160"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1536_864/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1536x864_video_type_1(audio: str, qpp: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 60
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 192"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1536_864/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1600x900_video_type_5(fps: str, audio: str, qpp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 24
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 96"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1600_900/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1920x1080_video_type_4(qpp: str, audio: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 30
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 96"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1920_1080/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1920x1080_video_type_3(fps: str, audio: str, qpp: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 45
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 128"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1920_1080/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1920x1080_video_type_2(qpp: str, audio: str, fps: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 48
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 160"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1920_1080/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_to_1920x1080_video_type_1(fps: str, qpp: str, audio: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call exemplify using the **params list:**
		
		1. **FPS** - Frames Per Second - Using 60
		2. **QPP** - Pixel quality percentage - Using 100%
		3. **Audio Bitrate** - Audio Quality - Using 192"
    
    """
    url = f"https://bitrate-calculator.p.rapidapi.com/1920_1080/{fps}/{qpp}/{audio}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitrate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

