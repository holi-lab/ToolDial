import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hashtag_challenge_posts(cursor: int=None, challenge_id: str='42164', challenge_name: str='foryou', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Hashtag/Challenge Posts By "challenge_name" or "challenge_id",
		and the endpoint will return 35 posts by 'cursor', with post data including the author data."
    cursor: put the cursor ther returned from the server in your Url param to get the next posts/items
        challenge_id: if u have the challenge_id  already use \\\\\\\\\\\\\\\"challenge_id\\\\\\\\\\\\\\\" param insted,
it will return data faster then the \\\\\\\\\\\\\\\"challenge_name\\\\\\\\\\\\\\\" param.
        challenge_name: if u have the challenge_id already use \\\\\\\\\\\\\\\"challenge_id\\\\\\\\\\\\\\\" param insted,
it will return data faster then the \\\\\\\\\\\\\\\"challenge_name\\\\\\\\\\\\\\\" param.
        
    """
    url = f"https://tiktok87.p.rapidapi.com/challenge_posts/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if challenge_id:
        querystring['challenge_id'] = challenge_id
    if challenge_name:
        querystring['challenge_name'] = challenge_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_challenge_info(challenge_name: str='foryou', challenge_id: str='42164', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Hashtag/Challenge Information By "challenge_name" or "challenge_id",
		and the endpoint will return Challenge Information.  //Notice: it will not return the challenge posts"
    
    """
    url = f"https://tiktok87.p.rapidapi.com/challenge_info/"
    querystring = {}
    if challenge_name:
        querystring['challenge_name'] = challenge_name
    if challenge_id:
        querystring['challenge_id'] = challenge_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_posts(secuid: str, cursor: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User Posts by 'user_id',
		and the endpoint will return 30 posts by cursor."
    secuid: provide [/user_info/] with unique_id and it will return 'secuid' in 'userInfo.user.secUid'.
        cursor: u can get this param value from the response to scroll.
        
    """
    url = f"https://tiktok87.p.rapidapi.com/user_posts/"
    querystring = {'secuid': secuid, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info_not_stable(unique_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User Information By "unique_id",
		and the endpoint will return user data, including => id, nickname, bio, secUid, avatar, statistics"
    
    """
    url = f"https://tiktok87.p.rapidapi.com/user_info/"
    querystring = {'unique_id': unique_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_posts(music_id: str, curor: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Music  Posts By "music_id",
		and the endpoint will return 25 posts by 'cursor', with post data including the author data."
    
    """
    url = f"https://tiktok87.p.rapidapi.com/music_posts/"
    querystring = {'music_id': music_id, }
    if curor:
        querystring['curor'] = curor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_info(music_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Music Information By "music_id",
		and the endpoint will return the Music's Information.  //Notice: it will not return the music posts"
    
    """
    url = f"https://tiktok87.p.rapidapi.com/music_info/"
    querystring = {'music_id': music_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info_by_id(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User Information By "user_id",
		and the endpoint will return user data, including => id, nickname, bio, secUid, avatar, statistics"
    
    """
    url = f"https://tiktok87.p.rapidapi.com/user_info_by_id/"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_comments(item_id: str, cursor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Posts CommentBy "challenge_name" or "challenge_id",
		and the endpoint will return Posts comments by 'cursor'."
    item_id: Post awme_id/item_id you will find it in [tiktok.com/@****/video/{item_d}]
        
    """
    url = f"https://tiktok87.p.rapidapi.com/post_comments/"
    querystring = {'item_id': item_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

