import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v2_user_followers(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User Followers"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/Followers/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_media(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User Media"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/UserMedia/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_users_by_rest_ids(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / Users By Rest IDs"
    ids: Users IDs (you can separate with commas)
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/UsersByRestIds/"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_by_screen_name(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User By Screen Name"
    username: Username
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/UserByScreenName/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_user_followers(cursor: str=None, is_id: str=None, username: str='elonmusk', count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / User Followers"
    cursor: Cursor token
        is_id: User ID
        username: Username

*Exactly one of **User ID** or **Username** must be provided.*
        count: Number of results. Max: `200`
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/Followers/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if is_id:
        querystring['id'] = is_id
    if username:
        querystring['username'] = username
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_user_following(is_id: str=None, count: int=20, username: str='elonmusk', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / User Following"
    is_id: User ID
        count: Number of results. Max: `200`
        username: Username

*Exactly one of **User ID** or **Username** must be provided.*
        cursor: Cursor token
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/Following/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if count:
        querystring['count'] = count
    if username:
        querystring['username'] = username
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_users(ids: str='44196397', usernames: str='BillGates,Google', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / Users"
    ids: User IDs

A comma separated list of User IDs. Up to `100` are allowed in a single request.
        usernames: User screen names

A comma separated list of Usernames. Up to `100` are allowed in a single request.
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/Users/"
    querystring = {}
    if ids:
        querystring['ids'] = ids
    if usernames:
        querystring['usernames'] = usernames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_tweet_retweeters_ids(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / Tweet Retweeters IDs"
    id: Tweet ID
        count: Number of results. Max: `100`
        cursor: Cursor token
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/TweetRetweetersIds/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_search_tweets(q: str, count: int=20, result_type: str=None, lang: str=None, until: str=None, max_id: str=None, since_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / Search Tweets"
    q: Search query
        count: Number of Tweet results. Max: `100`
        result_type: Result type. Default: `mixed`
        lang: Language of Tweet results
        until: Returns tweets created before the given date. Date should be formatted as YYYY-MM-DD. Keep in mind that the search index has a 7-day limit. In other words, no tweets will be found for a date older than one week.
        max_id: Returns results with an ID less than (that is, older than) or equal to the specified ID.
        since_id: Returns results with an ID greater than (that is, more recent than) the specified ID. There are limits to the number of Tweets which can be accessed through the API. If the limit of Tweets has occured since the since_id, the since_id will be forced to the oldest ID available.
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/SearchTweets/"
    querystring = {'q': q, }
    if count:
        querystring['count'] = count
    if result_type:
        querystring['result_type'] = result_type
    if lang:
        querystring['lang'] = lang
    if until:
        querystring['until'] = until
    if max_id:
        querystring['max_id'] = max_id
    if since_id:
        querystring['since_id'] = since_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_user_timeline(is_id: str=None, since_id: str=None, username: str='elonmusk', count: int=20, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / User Timeline"
    is_id: User ID
        since_id: Returns results with an ID greater than (that is, more recent than) the specified ID. There are limits to the number of Tweets that can be accessed through the API. If the limit of Tweets has occured since the since_id, the since_id will be forced to the oldest ID available.
        username: Username

*Exactly one of **User ID** or **Username** must be provided.*
        count: Number of results. Max: `200`
        max_id: Returns results with an ID less than (that is, older than) or equal to the specified ID.
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/UserTimeline/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if since_id:
        querystring['since_id'] = since_id
    if username:
        querystring['username'] = username
    if count:
        querystring['count'] = count
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_following(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User Following"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/Following/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_affiliates(is_id: int, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User Affiliates"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/UserAffiliates/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_likes(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User Likes"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/Likes/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_tweets(is_id: str, count: int=40, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User Tweets"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/UserTweets/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_by_rest_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User By Rest ID"
    id: User ID
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/UserByRestId/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_user_favorites(count: int=20, since_id: str=None, is_id: str=None, username: str='elonmusk', max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / User Favorites"
    count: Number of results. Max: `200`
        since_id: Returns results with an ID greater than (that is, more recent than) the specified ID. There are limits to the number of Tweets which can be accessed through the API. If the limit of Tweets has occured since the since_id, the since_id will be forced to the oldest ID available.	
        is_id: User ID
        username: Username

*Exactly one of **User ID** or **Username** must be provided.*
        max_id: Returns results with an ID less than (that is, older than) or equal to the specified ID.
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/UserFavorites/"
    querystring = {}
    if count:
        querystring['count'] = count
    if since_id:
        querystring['since_id'] = since_id
    if is_id:
        querystring['id'] = is_id
    if username:
        querystring['username'] = username
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_translate_tweet(is_id: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / Translate Tweet"
    id: Tweet ID
        language: Destination language
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/TranslateTweet/"
    querystring = {'id': is_id, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_tweet_retweeters(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / Tweet Retweeters"
    id: Tweet ID
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/Retweeters/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_tweet_favoriters(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / Tweet Favoriters"
    id: Tweet ID
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/Favoriters/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_tweet_detail_conversation(is_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / Tweet Detail & Conversation"
    id: Tweet ID
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/TweetDetail/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_tweets(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / Tweets"
    ids: Tweet IDs

A comma separated list of Tweet IDs. Up to `100` are allowed in a single request.
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/Tweets/"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_user_following_ids(cursor: str=None, username: str='elonmusk', is_id: str=None, count: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / User Following IDs"
    cursor: Cursor token
        username: Username

*Exactly one of **User ID** or **Username** must be provided.*
        is_id: User ID
        count: Number of results. Max: `5000`
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/FollowingIds/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if username:
        querystring['username'] = username
    if is_id:
        querystring['id'] = is_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_user_followers_ids(is_id: str=None, count: int=500, username: str='elonmusk', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / User Followers IDs"
    is_id: User ID
        count: Number of results. Max: `5000`
        username: Username

*Exactly one of **User ID** or **Username** must be provided.*
        cursor: Cursor token
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/FollowersIds/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if count:
        querystring['count'] = count
    if username:
        querystring['username'] = username
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_translate_profile(language: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v1.1 / Translate Profile"
    language: Destination language
        id: User ID
        
    """
    url = f"https://twitter135.p.rapidapi.com/v1.1/TranslateProfile/"
    querystring = {'language': language, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tweets(is_id: str, count: int=40, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Tweets"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/UserTweets/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
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
    url = f"https://twitter135.p.rapidapi.com/UsersByRestIds/"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Following"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/Following/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
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
    url = f"https://twitter135.p.rapidapi.com/UserByRestId/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
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
    url = f"https://twitter135.p.rapidapi.com/Retweeters/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_favoriters(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tweet Favoriters"
    id: Tweet ID
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/Favoriters/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_media(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Media"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/UserMedia/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
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
    url = f"https://twitter135.p.rapidapi.com/UserByScreenName/"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers(is_id: str, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Followers"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        cursor: Cursor for other results
        count: Number of results
        
    """
    url = f"https://twitter135.p.rapidapi.com/Followers/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_likes(is_id: str, count: int=20, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Likes"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/Likes/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
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
    url = f"https://twitter135.p.rapidapi.com/UserTweetsAndReplies/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_user_tweets_replies(is_id: str, count: int=40, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "v2 / User Tweets & Replies"
    id: User ID

Use the `User By Screen Name` endpoint to find the ID from a username.
        count: Number of results
        cursor: Cursor for other results
        
    """
    url = f"https://twitter135.p.rapidapi.com/v2/UserTweetsAndReplies/"
    querystring = {'id': is_id, }
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
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
    url = f"https://twitter135.p.rapidapi.com/TweetDetail/"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, cursor: str=None, count: int=20, tweet_search_mode: str=None, result_filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    q: Search query

You can use advanced search queries. E.g. `dogecoin (from:elonmusk)` Check out for more information: [https://twitter.com/search-advanced](https://twitter.com/search-advanced)
        cursor: Cursor for other results
        count: Number of Tweet results
        tweet_search_mode: Popular (default) or latest (live) tweets
        result_filter: Result filter
        
    """
    url = f"https://twitter135.p.rapidapi.com/Search/"
    querystring = {'q': q, }
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    if tweet_search_mode:
        querystring['tweet_search_mode'] = tweet_search_mode
    if result_filter:
        querystring['result_filter'] = result_filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
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
    url = f"https://twitter135.p.rapidapi.com/AutoComplete/"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

