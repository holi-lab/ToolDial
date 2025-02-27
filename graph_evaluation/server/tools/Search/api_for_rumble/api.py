import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def channel_page(channel: str, date: str='this-month', page: str='2', duration: str='short', sort: str='views', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel info - title, subscriber count, background image, thumbnail, verified, array of channel's videos, each containing: verified, title, link, image, channel, channel url, duration, views, time of publishing, etc.
		Optionally, add query parameters for sort, date, duration, page"
    channel: The channel as it appears on https://rumble.com/c/[CHANNEL]
        
    """
    url = f"https://api-for-rumble.p.rapidapi.com/c/{channel}"
    querystring = {}
    if date:
        querystring['date'] = date
    if page:
        querystring['page'] = page
    if duration:
        querystring['duration'] = duration
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-for-rumble.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def homepage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Info from Rumble homepage - top videos, editor picks, news, viral, and other categories"
    
    """
    url = f"https://api-for-rumble.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-for-rumble.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_videos(q: str, date: str='this-month', sort: str='views', duration: str='long', license: str='now', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All videos page. Array of video results, each containing: title, link, image, channel, channel url, duration, views, rumbles (if available), verified (if true), time.
		Add the keyword as a query parameter (?q=food).
		Optionally, add query parameters for sort, date, duration, license, page"
    
    """
    url = f"https://api-for-rumble.p.rapidapi.com/videos"
    querystring = {'q': q, }
    if date:
        querystring['date'] = date
    if sort:
        querystring['sort'] = sort
    if duration:
        querystring['duration'] = duration
    if license:
        querystring['license'] = license
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-for-rumble.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_search(q: str, page: str='2', duration: str='short', date: str='this-month', license: str='now', sort: str='views', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video search by keyword. Array of video results, each containing: title, link, image, channel, channel url, duration, views, time of publishing, etc.
		Add the keyword as a query parameter (?q=dogs).
		Optionally, add query parameters for sort, date, duration, license, page"
    
    """
    url = f"https://api-for-rumble.p.rapidapi.com/search/video"
    querystring = {'q': q, }
    if page:
        querystring['page'] = page
    if duration:
        querystring['duration'] = duration
    if date:
        querystring['date'] = date
    if license:
        querystring['license'] = license
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-for-rumble.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_search(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Channel search by keyword. Array of channel results, each containing: title, link, subscribers, verified (if true).
		Add the keyword as a query parameter (?q=food)"
    
    """
    url = f"https://api-for-rumble.p.rapidapi.com/search/channel"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-for-rumble.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_page(video: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Info from video page - title, channel, channel url, verified, publish date, views, subscribers, rumbles.
		In addition, an array with related videos.
		Use the html filename directly as path parameter (e.g. v30032-cat-scared-of-cucumber.html)"
    
    """
    url = f"https://api-for-rumble.p.rapidapi.com/{video}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-for-rumble.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def editor_picks(page: str='2', duration: str='short', license: str='now', sort: str='views', date: str='this-month', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Editor picks, each containing: verified, title, link, image, channel, channel url, duration, views, time of publishing, etc.
		Optionally, add query parameters for sort, date, duration, license, page"
    
    """
    url = f"https://api-for-rumble.p.rapidapi.com/editor-picks"
    querystring = {}
    if page:
        querystring['page'] = page
    if duration:
        querystring['duration'] = duration
    if license:
        querystring['license'] = license
    if sort:
        querystring['sort'] = sort
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-for-rumble.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

