import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def channel_profile_detail(channel_id: str, region: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel profile detail"
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/channel/{channel_id}/profile"
    querystring = {}
    if region:
        querystring['region'] = region
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_api(keyword: str, region: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos, playlist, channel."
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/search/{keyword}"
    querystring = {}
    if region:
        querystring['region'] = region
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_related_channels(channel_id: str, region: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related channels by channel id"
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/channel/{channel_id}/channels"
    querystring = {}
    if region:
        querystring['region'] = region
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_videos(channel_id: str, region: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos by channel id"
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/channel/{channel_id}/videos"
    querystring = {}
    if region:
        querystring['region'] = region
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_community_posts(channel_id: str, continuation: str='Egljb21tdW5pdHm4AQCSAwCqAxwKGFEwNU1UakEyVVVkRlRVUkdPRnAzUWc9PSgK8gYJCgdKAKIBAggB', hl: str='en', region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get community posts by channel id"
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/channel/{channel_id}/community"
    querystring = {}
    if continuation:
        querystring['continuation'] = continuation
    if hl:
        querystring['hl'] = hl
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_playlist(channel_id: str, continuation: str='4qmFsgJ4EhhVQ1dKMmxXTnViQXJIV21mM0ZJSGJmY1EaKkVnbHdiR0Y1YkdsemRITVlCQ0FCTUFFNEFlb0RCME5uVGtSUmFsRSUzRJoCL2Jyb3dzZS1mZWVkVUNXSjJsV051YkFySFdtZjNGSUhiZmNRcGxheWxpc3RzMTA0', region: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get playlists by channel id"
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/channel/{channel_id}/playlists"
    querystring = {}
    if continuation:
        querystring['continuation'] = continuation
    if region:
        querystring['region'] = region
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channel_details(channel_id: str, region: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channel details"
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/channel/{channel_id}"
    querystring = {}
    if region:
        querystring['region'] = region
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_comments(video_id: str, continuation: str='EkMSCzRzOENLdE5CUWdByAEA4AEBogINKP___________wFAAMICHQgEGhdodHRwczovL3d3dy55b3V0dWJlLmNvbSIAGAYywgIKqgJnZXRfcmFua2VkX3N0cmVhbXMtLUNxWUJDSUFFRlJlMzBUZ2Ftd0VLbGdFSTJGOFFnQVFZQnlLTEFSdjJid3lhNFVyQV9PZVRRTW1FWUVyR29NZXhjd3NyVklZNGhqWUlQR1llOGxVeDFTZVJlTFhjV2lvVkxaZUFDbmFoOS1rMEItbUlQSWJpbmFoTWxpY0JSMlQ3RlUxQUMxQTVlQ2loRE92RTlBNnVsVlVjOUZrcUVITWJpOFkya1BJZHZ1ZC1JRFhObl9DN2xiNmVUSUhBbmRNQVVxVmJMdTNjdUhFQ2hDaW9xaS1KNi1oSDJvSzdFaTBRRkJJRkNJZ2dHQUFTQlFpR0lCZ0FFZ2NJaFNBUUZCZ0JFZ1VJaHlBWUFCSUZDSWtnR0FBWUFBIhEiCzRzOENLdE5CUWdBMAB4ASgo', region: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get YouTube video comments."
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/video/{video_id}/comments"
    querystring = {}
    if continuation:
        querystring['continuation'] = continuation
    if region:
        querystring['region'] = region
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_details(video_id: str, hl: str='en', region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get YouTube video details."
    
    """
    url = f"https://youtube-api31.p.rapidapi.com/youtube/v1/video/{video_id}"
    querystring = {}
    if hl:
        querystring['hl'] = hl
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-api31.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

