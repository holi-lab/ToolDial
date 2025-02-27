import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def videos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all videos in the library"
    
    """
    url = f"https://anssi-flowplayer-drive-v1.p.rapidapi.com/videos"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anssi-flowplayer-drive-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_video(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows the specified video, with embedded encodings"
    
    """
    url = f"https://anssi-flowplayer-drive-v1.p.rapidapi.com/videos/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anssi-flowplayer-drive-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_account(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows the account"
    
    """
    url = f"https://anssi-flowplayer-drive-v1.p.rapidapi.com/account"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anssi-flowplayer-drive-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traffic(is_id: str, start: str='start time', end: str=None, grouping: str='hour', type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get traffic statistics for the specified video."
    start: Optional start value of the time range. Format yyyy-mm-dd, for example 2013-01-01. Default value is dependent on the specified grouping.
        end: Optional end value of the time range. Format yyyy-mm-dd. Defaults to yesterday.
        grouping: One of 'hour', 'day', 'week', 'month'. Specifies the granularity of the results. For example, in a day grouping one value is returned for each day of the time range. In a hour grouping one value is returned for every hour.
        type: Either 'views' or 'bytes'. Specifies if the results should be view counts or bytes. Defaults to 'views'.
        
    """
    url = f"https://anssi-flowplayer-drive-v1.p.rapidapi.com/videos/{is_id}/traffic"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    if grouping:
        querystring['grouping'] = grouping
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anssi-flowplayer-drive-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retention(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get retention data for the specified video. Retention data answers following questions: How long do viewers spend watching your videos? At what point do they lose interest or tune out and go elsewhere? The returned data tells the percentage (and count) of viewers that watched the video up to a given time in the video's timeline."
    
    """
    url = f"https://anssi-flowplayer-drive-v1.p.rapidapi.com/videos/{is_id}/retention"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anssi-flowplayer-drive-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

