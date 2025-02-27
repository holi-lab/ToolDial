import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_latest(query: str, twttr_proxy: str=None, twttr_session: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search latest results"
    twttr_session: Use login endpoint to get session token.
        
    """
    url = f"https://twttrapi.p.rapidapi.com/search-latest"
    querystring = {'query': query, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_top(query: str, twttr_session: str=None, twttr_proxy: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search top results"
    twttr_session: Use login endpoint to get session token.
        
    """
    url = f"https://twttrapi.p.rapidapi.com/search-top"
    querystring = {'query': query, }
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_suggestions(query: str, twttr_session: str=None, twttr_proxy: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search suggestions"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/search-suggestions"
    querystring = {'query': query, }
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocked_users(twttr_session: str, twttr_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show blocked users."
    
    """
    url = f"https://twttrapi.p.rapidapi.com/blocked-users"
    querystring = {'twttr-session': twttr_session, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def friendships_incoming(twttr_session: str, twttr_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show incoming friendships."
    
    """
    url = f"https://twttrapi.p.rapidapi.com/friendships-incoming"
    querystring = {'twttr-session': twttr_session, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_notifications(type: str, twttr_session: str=None, twttr_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get account notifications."
    type: type: all, verified, mentions
        
    """
    url = f"https://twttrapi.p.rapidapi.com/notifications"
    querystring = {'type': type, }
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_bookmarks(twttr_session: str, twttr_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all bookmarks."
    
    """
    url = f"https://twttrapi.p.rapidapi.com/all-bookmarks"
    querystring = {'twttr-session': twttr_session, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmark_collection(twttr_session: str, collection_id: str, twttr_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get bookmark collection."
    
    """
    url = f"https://twttrapi.p.rapidapi.com/bookmark-collection"
    querystring = {'twttr-session': twttr_session, 'collection_id': collection_id, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmark_collections(twttr_session: str, twttr_proxy: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get bookmark collections."
    
    """
    url = f"https://twttrapi.p.rapidapi.com/bookmark-collections"
    querystring = {'twttr-session': twttr_session, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_videos(query: str, twttr_proxy: str=None, twttr_session: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos"
    twttr_session: Use login endpoint to get session token.
        
    """
    url = f"https://twttrapi.p.rapidapi.com/search-videos"
    querystring = {'query': query, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers(twttr_proxy: str=None, twttr_session: str=None, username: str='elonmusk', user_id: str=None, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followers"
    twttr_session: Use login endpoint to get session token.
        count: Available for paid users only - Default is 20 (max is 100)
        
    """
    url = f"https://twttrapi.p.rapidapi.com/user-followers"
    querystring = {}
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def for_you_timeline(twttr_session: str, twttr_proxy: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the "For You" timeline"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/for-you-timeline"
    querystring = {'twttr-session': twttr_session, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_likes(twttr_session: str=None, twttr_proxy: str=None, user_id: str=None, username: str='elonmusk', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user's likes"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/user-likes"
    querystring = {}
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if user_id:
        querystring['user_id'] = user_id
    if username:
        querystring['username'] = username
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following(twttr_proxy: str=None, twttr_session: str=None, user_id: str=None, username: str='elonmusk', cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user following"
    twttr_session: Use login endpoint to get session token.
        count: Available for paid users only - Default is 20 (max is 100)
        
    """
    url = f"https://twttrapi.p.rapidapi.com/user-following"
    querystring = {}
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if user_id:
        querystring['user_id'] = user_id
    if username:
        querystring['username'] = username
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user(username: str, twttr_proxy: str=None, twttr_session: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user information"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/get-user"
    querystring = {'username': username, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_by_id(user_id: str, twttr_session: str=None, twttr_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user information by user id"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/get-user-by-id"
    querystring = {'user_id': user_id, }
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def following_timeline(twttr_session: str, twttr_proxy: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the "Following" timeline"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/following-timeline"
    querystring = {'twttr-session': twttr_session, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tweet(tweet_id: str, twttr_proxy: str=None, twttr_session: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a tweet"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/get-tweet"
    querystring = {'tweet_id': tweet_id, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_media(twttr_proxy: str=None, twttr_session: str=None, username: str='elonmusk', user_id: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user's images"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/user-media"
    querystring = {}
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tweets(twttr_proxy: str=None, twttr_session: str=None, cursor: str=None, user_id: str=None, username: str='elonmusk', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user's tweets"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/user-tweets"
    querystring = {}
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if cursor:
        querystring['cursor'] = cursor
    if user_id:
        querystring['user_id'] = user_id
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_dm_conversation(twttr_session: str, twttr_proxy: str=None, user_id: str=None, max_id: str=None, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get single conversation messages"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/get-dm-conversation"
    querystring = {'twttr-session': twttr_session, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if user_id:
        querystring['user_id'] = user_id
    if max_id:
        querystring['max_id'] = max_id
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_dm_conversations(twttr_session: str, twttr_proxy: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get direct message conversations and messages"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/get-dm-conversations"
    querystring = {'twttr-session': twttr_session, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_images(query: str, twttr_proxy: str=None, twttr_session: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search images"
    twttr_session: Use login endpoint to get session token.
        
    """
    url = f"https://twttrapi.p.rapidapi.com/search-images"
    querystring = {'query': query, }
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(query: str, twttr_session: str=None, twttr_proxy: str=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search users"
    twttr_session: Use login endpoint to get session token.
        
    """
    url = f"https://twttrapi.p.rapidapi.com/search-users"
    querystring = {'query': query, }
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tweet_conversation(tweet_id: str, twttr_session: str=None, twttr_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a tweet conversation"
    
    """
    url = f"https://twttrapi.p.rapidapi.com/get-tweet-conversation"
    querystring = {'tweet_id': tweet_id, }
    if twttr_session:
        querystring['twttr-session'] = twttr_session
    if twttr_proxy:
        querystring['twttr-proxy'] = twttr_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

