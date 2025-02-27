import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def instagram_get_media_info(media_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get any media info from instagram"
    media_id: Url parameters on instagram post.

For example, media id is CbxGLCFgyAa for url  below
https://www.instagram.com/p/CbxGLCFgyAa/
        
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/instagram/mediaInfo"
    querystring = {'media_id': media_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_get_video_comments(media_id: str='7A1q7v4btbk', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments of video from YouTube"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/youtube/mediaComments"
    querystring = {}
    if media_id:
        querystring['media_id'] = media_id
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_channel_info(channel_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel info from YouTube"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/youtube/channelInfo"
    querystring = {'channel_id': channel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def youtube_get_media_info(media_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video info from YouTube"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/youtube/mediaInfo"
    querystring = {'media_id': media_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter_get_replies(media_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tweet replies from twitter. This endpoint doesn't contains nested replies"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/twitter-uo/replies"
    querystring = {'media_id': media_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter_get_user_info(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user info from Twitter"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/twitter/userInfo/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter_get_retweets(media_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get retweets from any tweet on Twitter"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/twitter-uo/retweeters"
    querystring = {'media_id': media_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tiktok_get_user_info(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get any user info from TikTok"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/instagram/userInfo"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tiktok_get_media_comments(media_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments from any TikTok media"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/tiktok/mediaComments"
    querystring = {'media_id': media_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tiktok_get_media_info(media_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media info from TikTok"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/tiktok/mediaInfo"
    querystring = {'media_id': media_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instagram_get_user_medias(user_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of media from any Instagram profile"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/instagram/getMedias"
    querystring = {'user_id': user_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instagram_get_user_s_following_list(user_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get following lists of any Instagram users"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/instagram/userFollowing"
    querystring = {'user_id': user_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instagram_get_user_s_stories(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stories from any Instagram user"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/instagram/getStories"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instagram_get_user_info(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get any user info from Instagram"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/instagram/userInfo"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instagram_get_likes(media_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get likers from any Instagram media"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/instagram/mediaLikes"
    querystring = {'media_id': media_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instagram_get_media_comments(media_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments from any media on Instagram"
    
    """
    url = f"https://simpliers.p.rapidapi.com/api/get/instagram/mediaComments"
    querystring = {'media_id': media_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simpliers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

