import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user_followers(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Followers"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/Followers/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Following"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/Following/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_likes(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Likes"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/Likes/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_media(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Media"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/UserMedia/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tweets_replies(is_id: str, cursor: str=None, count: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Tweets & Replies"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/UserTweetsAndReplies/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tweets(is_id: str, cursor: str=None, count: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Tweets"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/UserTweets/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_by_rest_ids(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Users By Rest IDs"
    ids: Users IDs (you can separate with commas)
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/UsersByRestIds/"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_by_rest_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User By Rest ID"
    id: User ID
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/UserByRestId/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_by_screen_name(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User By Screen Name"
    username: Username
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/UserByScreenName/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_retweeters(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tweet Retweeters"
    id: Tweet ID
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/Retweeters/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_favoriters(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tweet Favoriters"
    id: Tweet ID
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/Favoriters/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_detail_conversation(is_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tweet Detail & Conversation"
    id: Tweet ID
        cursor: Cursor for other results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/TweetDetail/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, tweet_search_mode: str=None, result_filter: str=None, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    q: Search query

You can use advanced search queries. E.g. `dogecoin (from:elonmusk)` Check out for more information: [https://twitter.com/search-advanced](https://twitter.com/search-advanced)
        tweet_search_mode: Popular (default) or latest (live) tweets
        result_filter: Result filter
        count: Number of Tweet results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/Search/"
    querystring = {'q': q, }
    if tweet_search_mode:
        querystring['tweet_search_mode'] = tweet_search_mode
    if result_filter:
        querystring['result_filter'] = result_filter
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Auto Complete"
    q: Search query
        
    """
    url = f"https://twitter-v23.p.rapidapi.com/AutoComplete/"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-v23.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

