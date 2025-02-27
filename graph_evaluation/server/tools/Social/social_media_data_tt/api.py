import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def video_post_comments_v2(video_id: str, limit: int=50, max_cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post comments. V2 returns much more data then the old endpoint"
    video_id: Where to get vide id value?

For example in this video /@INFLUENCER/video/6818009093052189958 the id will be **6818009093052189958**
        limit: Number of records to return:

- Default is 50
- Maximum is 150

        max_cursor: Pagination cursor
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/post/comments/v2"
    querystring = {'video_id': video_id, }
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_metadata(music: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music metadata
		
		Basic metadata: number of posts,  direct url to the song(MP3) and etc"
    music: Music url
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/music"
    querystring = {'music': music, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_feed_video_posts(is_id: str, max_cursor: str=None, limit: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current music feed. 
		
		- Endpoint will return an array of objects with very useful metadata. 
		- Direct urls to the video , statistics and more."
    id: Music id

For example: 6823045620027099910
        max_cursor: Pagination cursor. 
To get more videos, paste here **max_cursor** value that you have received in previous request response.
        limit: Limit the output number of records. 

- Default is 20
- Max number is 100

        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/music/feed"
    querystring = {'id': is_id, }
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_post_comments(video_id: str, max_cursor: int=0, limit: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get video post comments."
    video_id: Where to get vide id value?

For example in this video /@INFLUENCER/video/6818009093052189958 the id will be **6818009093052189958**
        max_cursor: Pagination cursor
        limit: Number of records to return:

- Default is 40
- Maximum is 400

        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/post/comments"
    querystring = {'video_id': video_id, }
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_feed_video_posts_v2(limit: int=None, max_cursor: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending feed V2. 
		
		V2 - returns more data then older version of the endpoint, video without watermark and etc
		
		- Endpoint will return an array of objects with very useful metadata. 
		- Direct urls to the video , statistics and more."
    limit: Limit the output number of records. 

- Default is 12
- Max number is 12

        max_cursor: Pagination cursor. 
To get next batch of videos, paste here **max_cursor** value that you have received in previous request response.
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/trending/feed/v2"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_user_search(keyword: str, skip: int=None, limit: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for influencers by the keyword. 
		
		- Very useful data will be returned in the response, if user is verified or not, users country and more"
    keyword: Search keyword. For example: amazon
        skip: Skip N number of records in search result:

- Each response will have a value **has_more** if it is true then you can send next response with the same keyword but for example set **skip** value to 100
- It depends from the keyword it self, some keyword can find thousands of users
        limit: Limit the output number of records:

- Default is 10
- Maximum is 30

        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/user/search"
    querystring = {'keyword': keyword, }
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers_list(limit: int=40, max_cursor: int=0, username: str='amazon', sec_uid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followers:
		
		- Before testing don't forget to fill out the username **OR** sec_uid inputs"
    limit: Number of records to return:

- Default is 100
- Maximum is 200

        max_cursor: Pagination cursor. 
To get next batch of followers, paste here **max_cursor** value that you have received in previous request response.
        username: The influencer username. For example: **amazon**

- **NOTE:** By using **sec_uid** instead of the **username** request will be executed faster
- To use **sec_uid** use input field **BELOW**
        sec_uid: **Type** of a user id

**NOTE:** By using **sec_uid**, request will be executed faster then if you will use username

**NOTE:** **sec_uid** is not regular user id

**NOTE:** **sec_uid** can be obtained from the **User Information** endpoint

**NOTE:** **sec_uid** example: MS4wLjABAAAAv7iSuuXDJGDvJkmH_vz1qkDZYo1apxgzaxdBSeIuPiM
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/user/follower/list"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if username:
        querystring['username'] = username
    if sec_uid:
        querystring['sec_uid'] = sec_uid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followings_list(max_cursor: int=0, sec_uid: str=None, username: str='amazon', limit: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followings:
		
		- Before testing don't forget to fill out the username **OR** sec_uid inputs"
    max_cursor: Pagination cursor. 
To get next batch of followers, paste here **max_cursor** value that you have received in previous request response.
        sec_uid: **Type** of a user id

**NOTE:** By using **sec_uid**, request will be executed faster then if you will use username

**NOTE:** **sec_uid** is not regular user id

**NOTE:** **sec_uid** can be obtained from the **User Information** endpoint

**NOTE:** **sec_uid** example: MS4wLjABAAAAv7iSuuXDJGDvJkmH_vz1qkDZYo1apxgzaxdBSeIuPiM
        username: The influencer username. For example: **amazon**

- **NOTE:** By using **sec_uid** instead of the **username** request will be executed faster
- To use **sec_uid** use input field **BELOW**
        limit: Number of records to return:

- Default is 100
- Maximum is 200

        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/user/following/list"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if sec_uid:
        querystring['sec_uid'] = sec_uid
    if username:
        querystring['username'] = username
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_metadata_information(username: str, fresh: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user metadata. Number of followers, followings , avatar url, description and more"
    username: The influencer username. For example: **amazon**
        fresh: Force to return fresh data(not cached)
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/user"
    querystring = {'username': username, }
    if fresh:
        querystring['fresh'] = fresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_feed_video_posts(username: str='amazon', limit: int=None, max_cursor: int=None, sec_uid: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current user feed. 
		
		- Before testing don't forget to fill out the username **OR** sec_uid inputs
		- Endpoint will return an array of objects with very useful metadata. 
		- Direct urls to the video , statistics and more."
    username: The influencer username. For example: **charlidamelio**

- **NOTE:** By using **sec_uid** instead of the **username** request will be executed faster
- To use **sec_uid** use input field **BELOW**
        limit: Limit the output number of records. 

- Default is 100
- Max number is 500

        max_cursor: Pagination cursor. 
To get more videos, paste here **max_cursor** value that you have received in previous request response.
        sec_uid: **NOTE:** By using **sec_uid**, request will be executed faster then if you will use username

**NOTE:** **sec_uid** can be obtained from the **User Information** endpoint

**NOTE:** **sec_uid** example: MS4wLjABAAAAv7iSuuXDJGDvJkmH_vz1qkDZYo1apxgzaxdBSeIuPiM
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/user/feed"
    querystring = {}
    if username:
        querystring['username'] = username
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if sec_uid:
        querystring['sec_uid'] = sec_uid
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_feed_video_posts_v2(limit: int=None, max_cursor: int=None, username: str='tiktok', sec_uid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user feed V2
		
		V2 - returns more data then older version of the endpoint"
    limit: Limit the output number of records. 

- Default is 30
- Max number is 30

        max_cursor: Pagination cursor. 
To get more videos, paste here **max_cursor** value that you have received in previous request response.
        username: The influencer username. For example: **charlidamelio**

        sec_uid: **NOTE:** By using **sec_uid**, request will be executed faster then if you will use username

**NOTE:** **sec_uid** can be obtained from the **User Information** endpoint

**NOTE:** **sec_uid** example: MS4wLjABAAAAv7iSuuXDJGDvJkmH_vz1qkDZYo1apxgzaxdBSeIuPiM
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/user/feed/v2"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if username:
        querystring['username'] = username
    if sec_uid:
        querystring['sec_uid'] = sec_uid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_hashtag_search(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for hashtags by keyword"
    
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/hashtag/search"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def direct_post_url(video: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get direct post url"
    
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/post/direct"
    querystring = {'video': video, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_feed_video_posts(limit: int=None, hashtag_id: str=None, max_cursor: str=None, name: str='summer', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current hashtag feed. 
		
		- Before testing don't forget to fill out the name **OR** hashtag_id inputs
		- Endpoint will return an array of objects with very useful metadata. 
		- Direct urls to the video , statistics and more."
    limit: Limit the output number of records. 

- Default is 100
- Max number is 500

        hashtag_id: **NOTE:** By using **hashtag_id**, request will be executed faster then if you will use hashtag name

**NOTE:** **hashtag_id** can be obtained from the **/live/hashtag** endpoint

**NOTE:** **hashtag_id** example: 4232322
        max_cursor: Pagination cursor. 
To get more videos, paste here **max_cursor** value that you have received in previous request response.
        name: Hashtag name. For example: **duett**

- **NOTE:** By using **hashtag_id** instead of the hashtag **name** request will be executed faster
- To use **hashtag_id** use input field **BELOW**
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/hashtag/feed"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if hashtag_id:
        querystring['hashtag_id'] = hashtag_id
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_metadata_information(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag metadata
		
		Basic informations as number of posts and etc"
    hashtag: Hashtag name. For example: **summer**
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/hashtag"
    querystring = {'hashtag': hashtag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_metadata_information_v2(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag metadata V2
		
		V2 - returns more data then older version of the endpoint"
    hashtag: Hashtag name. For example: **summer**
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/hashtag/v2"
    querystring = {'hashtag': hashtag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_feed_video_posts(limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current trending feed. 
		
		- Due to nature of this endpoint the **max_cursor** is not required. Each request can return different data by default
		- Endpoint will return an array of objects with very useful metadata. 
		- Direct urls to the video , statistics and more."
    limit: Limit the output number of records. 

- Default is 20
- Max number is 20

        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/trending/feed"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_feed_video_posts_v2(name: str='summer', limit: int=None, max_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag feed V2. 
		
		V2 - returns more data then older version of the endpoint, video without watermark and etc
		
		- Before testing don't forget to fill out the **name** query
		- Endpoint will return an array of objects with very useful metadata. 
		- Direct urls to the video , statistics and more."
    name: Hashtag name. For example: **duett**
        limit: Limit the output number of records. 

- Default is 20
- Max number is 20

        max_cursor: Pagination cursor. 
To get more videos, paste here **max_cursor** value that you have received in previous request response.
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/hashtag/feed/v2"
    querystring = {}
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_post_metadata(video: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get single post metadata"
    video: Post url
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/post/meta"
    querystring = {'video': video, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_post_metadata_v2(video: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get single post metadata V2
		
		V2 - returns more data then older version of the endpoint"
    video: Post url
        
    """
    url = f"https://social-media-data-tt.p.rapidapi.com/live/post/meta/v2"
    querystring = {'video': video, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-media-data-tt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

