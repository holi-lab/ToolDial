import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tweets(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1/tweets"
    ids: Tweet IDs

A comma separated list of Tweet IDs. Up to 100 are allowed in a single request.
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/v1.1/Tweets/"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_tweets(q: str, until: str=None, since_id: str=None, max_id: str=None, result_type: str=None, lang: str=None, count: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1/search-tweets"
    q: Search query
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/v1.1/SearchTweets/"
    querystring = {'q': q, }
    if until:
        querystring['until'] = until
    if since_id:
        querystring['since_id'] = since_id
    if max_id:
        querystring['max_id'] = max_id
    if result_type:
        querystring['result_type'] = result_type
    if lang:
        querystring['lang'] = lang
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_detail(is_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "tweet-detail"
    id: Tweet ID
        cursor: Cursor for other results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/TweetDetail/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_favoriters(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "tweet-favoriters"
    id: Tweet ID
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/Favoriters/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_retweeters(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "tweet-retweeters"
    id: Tweet ID
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/Retweeters/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_tweet_detail(is_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2/tweet-detail"
    id: Tweet ID
        cursor: Cursor for other results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/v2/TweetDetail/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_by_screen_name(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-by-screen-name"
    username: username
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/UserByScreenName/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_by_rest_ids(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "users-by-rest-ids"
    ids: Users IDs (you can separate with commas)
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/UsersByRestIds/"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tweets(is_id: str, cursor: str=None, count: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-tweets"
    id: User ID

Use the User By Screen Name endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/UserTweets/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_likes(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-likes"
    id: User ID

Use the User By Screen Name endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/Likes/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_media(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-media"
    id: User ID

Use the User By Screen Name endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/UserMedia/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-followers"
    id: User ID

Use the User By Screen Name endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/Followers/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tweets_and_replies(is_id: str, count: int=40, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-tweets-and-replies"
    id: User ID

Use the User By Screen Name endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/UserTweetsAndReplies/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "user-following"
    id: User ID

Use the User By Screen Name endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/Following/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_tweets_and_replies(is_id: str, count: int=40, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2/user-tweets-and-replies"
    id: User ID

Use the User By Screen Name endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/v2/UserTweetsAndReplies/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, result_filter: str=None, cursor: str=None, tweet_search_mode: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search"
    q: Search query

You can use advanced search queries. E.g. dogecoin (from:elonmusk) Check out for more information: https://twitter.com/search-advanced
        result_filter: Result filter
        cursor: Cursor for other results
        tweet_search_mode: Popular (default) or latest (live) tweets
        count: Number of Tweet results
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/Search/"
    querystring = {'q': q, }
    if result_filter:
        querystring['result_filter'] = result_filter
    if cursor:
        querystring['cursor'] = cursor
    if tweet_search_mode:
        querystring['tweet_search_mode'] = tweet_search_mode
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "auto-complete"
    q: Search query
        
    """
    url = f"https://twitter-data1.p.rapidapi.com/AutoComplete/"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

