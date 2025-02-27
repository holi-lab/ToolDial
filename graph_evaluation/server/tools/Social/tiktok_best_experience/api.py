import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def followings_by_user_id(user_id: str, min_time: int=None, max_time: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET user's followings by User ID (uid/id)."
    min_time: used for pagination, can be found in response to previous request
        max_time: used for pagination, can be found in response to previous request
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/user/id/{user_id}/followings"
    querystring = {}
    if min_time:
        querystring['min_time'] = min_time
    if max_time:
        querystring['max_time'] = max_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_s_challenge_data_by_id(challenge_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a hashtag's (challenge)  Data by its ID."
    
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/challenge/id/{challenge_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_s_feed_by_id(music_id: str, max_cursor: int=18, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a music's Feed by its ID."
    max_cursor: If in a response you get parameter **has_more** equal to 1 then you also have **cursor** value  for a next set 
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/music/{music_id}/feed"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_s_data_by_id(music_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a music's Data by its ID."
    
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/music/{music_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_feed_by_id(user_id: str, max_cursor: int=1630342868000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a user's Feed by their ID (uid/id)."
    user_id: user's ID (also know as **userId**,  **authorId**, and **uid**)
        max_cursor: If in a response you get parameter **has_more** equal to 1 then you also have **max_cursor** value  for a next set 
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/user/id/{user_id}/feed"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_data_by_id(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a user's Data (including uniqueId [*username*]) by their ID (uid/id)."
    user_id: user's ID (also know as **userId**,  **authorId**, and **uid**)
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/user/id/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_by_id(video_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a Video by its ID."
    
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/video/{video_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers_by_user_id(user_id: str, max_time: int=None, min_time: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET user's followers by User ID (uid/id)."
    max_time: used for pagination, can be found in response to previous request
        min_time: used for pagination, can be found in response to previous request
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/user/id/{user_id}/followers"
    querystring = {}
    if max_time:
        querystring['max_time'] = max_time
    if min_time:
        querystring['min_time'] = min_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_videos_for_region(region: str, custom_cursor: int=7018021176124819456, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET Trending Videos for a **specific Region**. For now, supported regions are: **RU**, **US**, **GB**, **DE**, **FR**, **ES**, **AU**, **BR**, **CA**, **GR**, **IL**, **IT**, **MX**, **NL**, **PE**, **PL**, **SE**, **UA**, **VE**, **ID**, **SV**, **CR**, **TR**, **PT**, **TH**.
		If you need any other region please contact us via telegram."
    region: You should specify a region for Trending Videos.
For now, supported regions are: **RU**, **US**, **GB**, **DE**, **FR**, **ES**, **AU**, **BR**, **CA**, **GR**, **IL**, **IT**, **MX**, **NL**, **PE**, **PL**, **SE**, **UA**, **VE**, **ID**, **SV**, **CR**, **TR**, **PT**, **TH**.
If you need any other region please contact us via telegram.
        custom_cursor: In a response you have a **custom_cursor** parameter's value  for a next set 
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/trending/{region}"
    querystring = {}
    if custom_cursor:
        querystring['custom_cursor'] = custom_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_videos(custom_cursor: int=7018021176124819456, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET Trending Videos."
    
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/trending"
    querystring = {}
    if custom_cursor:
        querystring['custom_cursor'] = custom_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hashtags_challenges_by_query(query: str, max_cursor: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a list of  Hashtags (Challenges) by a Query (string occurrence)."
    max_cursor: If in a response you get parameter **has_more** equal to 1 then you also have **cursor** value  for a next set 
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/search/challenge/{query}"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_data_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a user's Data by their Username."
    
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/user/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_feed_by_username(username: str, max_cursor: str='1630342868000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a user's Feed by their Username."
    max_cursor: If in a response you get parameter **has_more** equal to 1 then you also have **max_cursor** value  for a next set 
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/user/{username}/feed"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users_by_query(query: str, max_cursor: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a list of Users by a Query (string occurrence)."
    query: Username 
        max_cursor: If in a response you get parameter **has_more** equal to 1 then you also have **cursor** value  for a next set 
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/search/user/{query}"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_s_challenge_feed_by_id(challenge_id: str, max_cursor: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a hashtag's (challenge) Feed by its ID."
    max_cursor: If in a response you get parameter **has_more** equal to 1 then you also have **cursor** value  for a next set 
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/challenge/id/{challenge_id}/feed"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_s_challenge_data_by_name(challenge_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a hashtag's (challenge)  Data by its Name."
    
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/challenge/{challenge_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_s_challenge_feed_by_name(challenge_name: str, max_cursor: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET a hashtag's (challenge) Feed by its Name."
    max_cursor: If in a response you get parameter **has_more** equal to 1 then you also have **cursor** value  for a next set 
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/challenge/{challenge_name}/feed"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_by_video_s_id(video_id: str, max_cursor: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET video's Comments by its ID."
    max_cursor: If in response you get parameter **hasMore** equal to 1 then you also have **cursor** value for a next set
        
    """
    url = f"https://tiktok-best-experience.p.rapidapi.com/comments/{video_id}"
    querystring = {}
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

