import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def initconfig(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get init config"
    
    """
    url = f"https://remote-commentator.p.rapidapi.com/publicApi/v2/initConfig"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "remote-commentator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_auido_track(eventid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get audio track"
    
    """
    url = f"https://remote-commentator.p.rapidapi.com/eventPreview/getAudioTrack/{eventid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "remote-commentator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def task_info(taskids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event info by Audio Mixer task ids
		
		**URL path Parameter:** 
		
		|  param | require   | type  |description |
		| ------------ | ------------ | ------------ | ------------ | 
		|  taskIds |  yes  |  string | task ids connected by "," |"
    
    """
    url = f"https://remote-commentator.p.rapidapi.com/am/taskInfo/{taskids}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "remote-commentator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_running_task_ids(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all running AM task ids, Limit 100"
    
    """
    url = f"https://remote-commentator.p.rapidapi.com/am/getRunningTaskIds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "remote-commentator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_setting_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user setting list
		
		**Response:** 
		
		> attention: the result is json object
		
		| **param**  | **require** | **type** |                      **description**                       |
		| :--------: | :---------: | :------: | :--------------------------------------------------------: |
		| maxBitrate |     No      |  int  | if it is null , you should get default value from media-service. |
		| cloudProxy |     No      |  boolean  | cloud proxy open/close. |"
    
    """
    url = f"https://remote-commentator.p.rapidapi.com/userSetting/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "remote-commentator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def input_and_audio_track(eventid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get input and input audio tarck info"
    
    """
    url = f"https://remote-commentator.p.rapidapi.com/eventInput/inputAndAudioTrack/{eventid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "remote-commentator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_local_sdi(rid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get R local SDI"
    
    """
    url = f"https://remote-commentator.p.rapidapi.com/receiver/getLocalSDI/{rid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "remote-commentator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

