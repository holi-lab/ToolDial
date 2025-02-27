import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def continuation_user_s_media(user_id: str, continuation_token: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the list of a user's medias"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/medias/continuation"
    querystring = {'user_id': user_id, 'continuation_token': continuation_token, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_media(user_id: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return a list of user's media given a user ID"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/medias"
    querystring = {'user_id': user_id, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_tweets(include_replies: bool=None, user_id: str='96479162', username: str='omarmhaimdat', limit: int=40, include_pinned: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return a list of user's tweets given a username"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/tweets"
    querystring = {}
    if include_replies:
        querystring['include_replies'] = include_replies
    if user_id:
        querystring['user_id'] = user_id
    if username:
        querystring['username'] = username
    if limit:
        querystring['limit'] = limit
    if include_pinned:
        querystring['include_pinned'] = include_pinned
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def available_locations_beta(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available locations"
    
    """
    url = f"https://twitter154.p.rapidapi.com/trends/available"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trends_near_a_location_beta(woeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top 50 trending topics for a specific id (woeid), if trending information is available for it."
    
    """
    url = f"https://twitter154.p.rapidapi.com/trends/"
    querystring = {'woeid': woeid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geo_search_beta(query: str, latitude: str='48.858093', longitude: str='2.294694', range: str='2', limit: str='50', language: str='en', section: str='top', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform Geo search"
    
    """
    url = f"https://twitter154.p.rapidapi.com/search/geo"
    querystring = {'query': query, }
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    if range:
        querystring['range'] = range
    if limit:
        querystring['limit'] = limit
    if language:
        querystring['language'] = language
    if section:
        querystring['section'] = section
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_user_retweets_continuation(tweet_id: str, continuation_token: str, limit: str='40', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the next  list of user who retweeted the tweet"
    
    """
    url = f"https://twitter154.p.rapidapi.com/tweet/retweets/continuation"
    querystring = {'tweet_id': tweet_id, 'continuation_token': continuation_token, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_user_retweets(tweet_id: str, limit: str='40', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of user who retweeted the tweet"
    
    """
    url = f"https://twitter154.p.rapidapi.com/tweet/retweets"
    querystring = {'tweet_id': tweet_id, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lists_tweets_continuation(list_id: str, continuation_token: str, limit: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the next list of tweets in a given Twitter list"
    
    """
    url = f"https://twitter154.p.rapidapi.com/lists/tweets"
    querystring = {'list_id': list_id, 'continuation_token': continuation_token, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lists_tweets(list_id: str, limit: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return a list of tweets in a given Twitter list"
    
    """
    url = f"https://twitter154.p.rapidapi.com/lists/tweets"
    querystring = {'list_id': list_id, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lists_details(list_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the public information a Twitter Lists"
    
    """
    url = f"https://twitter154.p.rapidapi.com/lists/details"
    querystring = {'list_id': list_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_details(tweet_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return general information about a tweet"
    
    """
    url = f"https://twitter154.p.rapidapi.com/tweet/details"
    querystring = {'tweet_id': tweet_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, section: str='top', min_replies: int=None, end_date: str=None, min_retweets: int=1, min_likes: int=1, start_date: str='2022-01-01', language: str='en', limit: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return search results"
    start_date: YYYY-MM-DD
        
    """
    url = f"https://twitter154.p.rapidapi.com/search/search"
    querystring = {'query': query, }
    if section:
        querystring['section'] = section
    if min_replies:
        querystring['min_replies'] = min_replies
    if end_date:
        querystring['end_date'] = end_date
    if min_retweets:
        querystring['min_retweets'] = min_retweets
    if min_likes:
        querystring['min_likes'] = min_likes
    if start_date:
        querystring['start_date'] = start_date
    if language:
        querystring['language'] = language
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details(username: str='omarmhaimdat', user_id: str='96479162', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the public information about a Twitter profile"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/details"
    querystring = {}
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_user_favoriters(tweet_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of user who favorited the tweet"
    
    """
    url = f"https://twitter154.p.rapidapi.com/tweet/favoriters"
    querystring = {'tweet_id': tweet_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_user_favoriters_continuation(tweet_id: str, continuation_token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the next list of user who favorited the tweet"
    
    """
    url = f"https://twitter154.p.rapidapi.com/tweet/favoriters/continuation"
    querystring = {'tweet_id': tweet_id, 'continuation_token': continuation_token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_replies(tweet_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of reply tweets"
    
    """
    url = f"https://twitter154.p.rapidapi.com/tweet/replies"
    querystring = {'tweet_id': tweet_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_followers(user_id: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return a list of user's followers given a user ID"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/followers"
    querystring = {'user_id': user_id, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_username(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a User's username given a user ID"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/id"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def continuation_user_s_likes(user_id: str, continuation_token: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the list of a user's Likes"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/likes/continuation"
    querystring = {'user_id': user_id, 'continuation_token': continuation_token, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_likes(user_id: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return a list of user's likes given a user ID"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/likes"
    querystring = {'user_id': user_id, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag(hashtag: str, limit: str='20', section: str='top', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return hashtag results"
    
    """
    url = f"https://twitter154.p.rapidapi.com/hashtag/hashtag"
    querystring = {'hashtag': hashtag, }
    if limit:
        querystring['limit'] = limit
    if section:
        querystring['section'] = section
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_continuation(hashtag: str, continuation_token: str, section: str='top', limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint return the next hashtag results"
    
    """
    url = f"https://twitter154.p.rapidapi.com/hashtag/hashtag/continuation"
    querystring = {'hashtag': hashtag, 'continuation_token': continuation_token, }
    if section:
        querystring['section'] = section
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_s_following(user_id: str, limit: int=40, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the list of following"
    
    """
    url = f"https://twitter154.p.rapidapi.com/user/following"
    querystring = {'user_id': user_id, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

