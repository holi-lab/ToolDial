import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_post_comments(video_id: str, fresh: int=0, limit: int=50, max_cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video post comments."
    video_id: Where to get vide id value?

For example in this video /@INFLUENCER/video/6818009093052189958 the id will be **6818009093052189958**
        fresh: By setting this query value to **1** you can force the API to return fresh data(not cached)
        limit: Number of records to return:

- Default is 50
- Maximum is 150

        max_cursor: Pagination cursor
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/post/comments/v2"
    querystring = {'video_id': video_id, }
    if fresh:
        querystring['fresh'] = fresh
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_metadata_information(username: str, fresh: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user metadata. Number of followers, followings , avatar url, description and more"
    username: TikTok username. For example: **amazon**
        fresh: By setting this query value to **1** you can force the API to return fresh data(not cached)
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/user"
    querystring = {'username': username, }
    if fresh:
        querystring['fresh'] = fresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers_list(fresh: int=0, sec_uid: str=None, max_cursor: str=None, limit: str=None, username: str='tiktok', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followers:
		
		- Before testing don't forget to fill out the username **OR** sec_uid inputs"
    fresh: By setting this query value to **1** you can force the API to return fresh data(not cached)
        sec_uid: **Type** of a user id

**NOTE:** By using **sec_uid**, request will be executed faster then if you will use username

**NOTE:** **sec_uid** is not regular user id

**NOTE:** **sec_uid** can be obtained from the **User Information** endpoint

**NOTE:** **sec_uid** example: MS4wLjABAAAAv7iSuuXDJGDvJkmH_vz1qkDZYo1apxgzaxdBSeIuPiM
        max_cursor: Pagination cursor. 
To get next batch of followers, paste here **max_cursor** value that you have received in previous request response.
        limit: Number of records to return:

- Default is 100
- Maximum is 100

        username: Tiktok username. For example: **amazon**

- **NOTE:** By using **sec_uid** instead of the **username** request will be executed faster
- To use **sec_uid** use input field **BELOW**
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/user/follower/list"
    querystring = {}
    if fresh:
        querystring['fresh'] = fresh
    if sec_uid:
        querystring['sec_uid'] = sec_uid
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if limit:
        querystring['limit'] = limit
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_metadata(music: str, fresh: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music metadata
		
		Basic metadata: number of posts,  direct url to the song(MP3) and etc"
    music: Tiktok music url. For example: **https://www.tiktok.com/music/Streets-x-Kiss-it-better-7090403288818584347**
        fresh: By setting this query value to **1** you can force the API to return fresh data(not cached)
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/music"
    querystring = {'music': music, }
    if fresh:
        querystring['fresh'] = fresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_feed_video_posts(is_id: str, fresh: int=0, limit: str=None, max_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current music feed. 
		
		- Endpoint will return an array of objects with very useful metadata. 
		- Direct urls to the video , statistics and more."
    id: Tiktok Music Id.

For example: https://www.tiktok.com/music/Streets-x-Kiss-it-better-7090403288818584347

**7090403288818584347** will be the required music id
        fresh: By setting this query value to **1** you can force the API to return fresh data(not cached)
        limit: Limit the output number of records. 

- Default is 20
- Max number is 100

        max_cursor: Pagination cursor. 
To get more videos, paste here **max_cursor** value that you have received in previous request response.
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/music/feed"
    querystring = {'id': is_id, }
    if fresh:
        querystring['fresh'] = fresh
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_video_without_the_watermark(fresh: int=0, video: str='https://www.tiktok.com/@charlidamelio/video/7137423965982592302', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get direct post url to the video without watermark"
    fresh: By setting this query value to **1** you can force the API to return fresh data(not cached)
        video: Tiktok video url. For example https://www.tiktok.com/@charlidamelio/video/7137423965982592302
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/post/direct"
    querystring = {}
    if fresh:
        querystring['fresh'] = fresh
    if video:
        querystring['video'] = video
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_post_metadata(video: str, fresh: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get single post metadata"
    video: TikTok video post url. For example: **https://www.tiktok.com/@charlidamelio/video/7137423965982592302**
        fresh: By setting this query value to **1** you can force to return fresh data(not cached)
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/post/meta/v2"
    querystring = {'video': video, }
    if fresh:
        querystring['fresh'] = fresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_metadata_information(hashtag: str, fresh: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag metadata"
    hashtag: Hashtag name. For example: **summer**
        fresh: By setting this query value to **1** you can force to return fresh data(not cached)
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/hashtag/v2"
    querystring = {'hashtag': hashtag, }
    if fresh:
        querystring['fresh'] = fresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_feed_video_posts(fresh: int=0, name: str='summer', limit: int=None, max_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag feed
		
		- Before testing don't forget to fill out the **name** query
		- Endpoint will return an array of objects with very useful metadata. 
		- Direct urls to the video , statistics and more."
    fresh: By setting this query value to **1** you can force to return fresh data(not cached)
        name: Hashtag name. For example: **duett**
        limit: Limit the output number of records. 

- Default is 20
- Max number is 20

        max_cursor: Pagination cursor. 
To get more videos, paste here **max_cursor** value that you have received in previous request response.
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/hashtag/feed/v2"
    querystring = {}
    if fresh:
        querystring['fresh'] = fresh
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_hashtag_search(keyword: str, fresh: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for hashtags by keyword"
    fresh: By setting this query value to **1** you can force to return fresh data(not cached)
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/hashtag/search"
    querystring = {'keyword': keyword, }
    if fresh:
        querystring['fresh'] = fresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_feed_video_posts(fresh: int=0, sec_uid: str=None, limit: str=None, username: str='tiktok', max_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user feed"
    fresh: By setting this query value to **1** you can force the API to return fresh data(not cached)
        sec_uid: **NOTE:** By using **sec_uid**, request will be executed faster then if you will use username

**NOTE:** **sec_uid** can be obtained from the **User Information** endpoint

**NOTE:** **sec_uid** example: MS4wLjABAAAAv7iSuuXDJGDvJkmH_vz1qkDZYo1apxgzaxdBSeIuPiM
        limit: Limit the output number of records. 

- Default is 30
- Max number is 30

        username: Tiktok username. For example: **charlidamelio**

        max_cursor: Pagination cursor. 
To get more videos, paste here **max_cursor** value that you have received in previous request response.
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/user/feed/v2"
    querystring = {}
    if fresh:
        querystring['fresh'] = fresh
    if sec_uid:
        querystring['sec_uid'] = sec_uid
    if limit:
        querystring['limit'] = limit
    if username:
        querystring['username'] = username
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followings_list(max_cursor: str=None, fresh: int=0, sec_uid: str=None, limit: str='40', username: str='tiktok', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followings:
		
		- Before testing don't forget to fill out the username **OR** sec_uid inputs"
    max_cursor: Pagination cursor. 
To get next batch of followers, paste here **max_cursor** value that you have received in previous request response.
        fresh: By setting this query value to **1** you can force the API to return fresh data(not cached)
        sec_uid: **Type** of a user id

**NOTE:** By using **sec_uid**, request will be executed faster then if you will use username

**NOTE:** **sec_uid** is not regular user id

**NOTE:** **sec_uid** can be obtained from the **User Information** endpoint

**NOTE:** **sec_uid** example: MS4wLjABAAAAv7iSuuXDJGDvJkmH_vz1qkDZYo1apxgzaxdBSeIuPiM
        limit: Number of records to return:

- Default is 100
- Maximum is 200

        username: The influencer username. For example: **amazon**

- **NOTE:** By using **sec_uid** instead of the **username** request will be executed faster
- To use **sec_uid** use input field **BELOW**
        
    """
    url = f"https://tiktok-data1.p.rapidapi.com/live/user/following/list"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if fresh:
        querystring['fresh'] = fresh
    if sec_uid:
        querystring['sec_uid'] = sec_uid
    if limit:
        querystring['limit'] = limit
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

