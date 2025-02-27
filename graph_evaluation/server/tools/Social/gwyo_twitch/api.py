import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_m3u8_links(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All resolutions m3u8 link URLs if the status is Online (by username or id)
		( return an empty JSON object if not currently LIVE )"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/m3u8/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_get_channel_followers_count(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Followers Count"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/followerscount/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_profile_pic_html_element(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Profile Pictures as a HTML Element with <img/> tag (using username or id)
		( Sizes: 150x150, 300x300, 600x600 )"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/userpic-img/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_profile_pic_url(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Profile Picture's URL (using username or id)
		( Sizes: 150x150, 300x300, 600x600 )"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/userpic/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stream_preview_url(userid: str, width: int=350, height: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Last Picture Preview's Url from a Live Stream
		( using optional width or width/height )
		( return an empty JSON object if not currently LIVE )"
    width: width ( in px ) between 144 and 2048
( default: 256px )
        height: height ( in px ) between 81 and 1152
auto adjust with the width if not specified
        
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/preview/{userid}/{width}/{height}"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stream_preview_as_html_element(userid: str, width: int=350, height: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Last Picture Preview as a HTML Element from a Live Stream
		( using optional width or width/height )
		( return an empty JSON object if not currently LIVE )"
    width: width ( in px ) between 144 and 2048
( default: 256px )
        height: height ( in px ) between 81 and 1152
auto adjust to the width if not specified
        
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/preview-img/{userid}/{width}/{height}"
    querystring = {}
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stream_game_name(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  Stream Game Name (by username or id)
		( return an empty JSON object if not currently LIVE )"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/game/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stream_mature_content_status(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Stream Mature Content Status (using username or id)
		( return an empty JSON object if not currently LIVE )"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/mature/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stream_number_of_viewers(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Stream Number of Viewers (using username or id)
		( return an empty JSON object if not currently LIVE )"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/viewers/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_active_stream_infos(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Stream Informations if the status is Online (by username or id)
		( return an empty JSON object if not currently LIVE )"
    userid: Username or Id can be used as UserId
        
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/stream/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stream_language(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Stream Language (using username or id)
		( return an empty JSON object if not currently LIVE )"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/lang/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_channel_infos(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Infos (by username or id)"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/user/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_description(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Description (using username or id)"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/description/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_total_views(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Cumulated Views since its creation (using username or id)"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/views/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_broadcaster_type(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Broadcaster Type (using username or id)"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/type/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stream_title(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Stream Title (using username or id)
		( return an empty JSON object if not currently LIVE )"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/title/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_channel_username_by_id(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Channel Username by Id"
    
    """
    url = f"https://gwyo-twitch.p.rapidapi.com/username/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gwyo-twitch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

