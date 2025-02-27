import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user_followings_by_userid(userid: str, cursor: str=None, count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all followings of a user. You can use the cursor and count to navigate around."
    userid: The userId of the user. You can get it by converting a username to the userId or by searching a user and extracting the restId.
        cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        count: The default count is 20. You can specify a custom one here.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/user/{userid}/followings"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers_by_userid(userid: str, count: str='20', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all followers of a user. You can use the cursor and count to navigate around."
    userid: The userId of the user. You can get it by converting a username to the userId or by searching a user and extracting the restId.
        count: The default count is 20. You can specify a custom one here.
        cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/user/{userid}/followers"
    querystring = {}
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweets_and_replies_by_userid(userid: str, count: str='20', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all tweets and replies ordered by most recent. You can use the cursor and count to navigate around."
    userid: The userId of the user. You can get it by converting a username to the userId or by searching a user and extracting the restId.
        count: The default count is 20. You can specify a custom one here.
        cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/user/{userid}/tweetsAndReplies"
    querystring = {}
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_by_userid(userid: str, cursor: str=None, count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all media items ordered by most recent. You can use the cursor and count to navigate around."
    userid: The userId of the user. You can get it by converting a username to the userId or by searching a user and extracting the restId.
        cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        count: The default count is 20. You can specify a custom one here.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/user/{userid}/media"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details_by_userid(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the prefered method of getting any information. Receive follower counts, images and more user related data with this endpoint."
    userid: The userId is mostly mentioned as restId.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/user/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_details_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive follower counts, images and more user related data with this endpoint."
    
    """
    url = f"https://twitter-aio.p.rapidapi.com/user/by/username/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweets_by_userid(userid: str, cursor: str=None, count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all tweets ordered by most recent. You can use the cursor and count to navigate around."
    userid: The userId of the user. You can get it by converting a username to the userId or by searching a user and extracting the restId.
        cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        count: The default count is 20. You can specify a custom one here.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/user/{userid}/tweets"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_likes_by_userid(userid: str, cursor: str=None, count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all likes of a user. You can use the cursor and count to navigate around."
    userid: The userId of the user. You can get it by converting a username to the userId or by searching a user and extracting the restId.
        cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        count: The default count is 20. You can specify a custom one here.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/user/{userid}/likes"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter_hashtag_emojis(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive a full list of all twitter custom emojis with the hashtag, image and more."
    
    """
    url = f"https://twitter-aio.p.rapidapi.com/misc/emojis"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_retweets(tweetid: str, count: str='20', cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all retweets for a tweet."
    count: The default count is 20. You can specify a custom one here.
        cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/tweet/{tweetid}/retweets"
    querystring = {}
    if count:
        querystring['count'] = count
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_likes(tweetid: str, cursor: str=None, count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all likes for a tweet."
    cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        count: The default count is 20. You can specify a custom one here.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/tweet/{tweetid}/favorites"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_details(tweetid: str, cursor: str=None, count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all tweet informations and comments by tweetId."
    cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        count: The default count is 20. You can specify a custom one here.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/tweet/{tweetid}"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_username_to_userid(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You'll need the userId to receive tweets, media and replies."
    
    """
    url = f"https://twitter-aio.p.rapidapi.com/username/to/id/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(searchterm: str, category: str=None, cursor: str=None, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With the search endpoint you can search all of twitter. You just need to provide a search term."
    cursor: At the end of the entries array you'll find two objects with the type TimelineTimelineCursor. There is one cursor to go up (Top) and one for going down (bottom) the list. You just need to provide the value as the cursor parameter to get the next page of the pagination.
        count: The default count is 20. You can specify a custom one here.
        
    """
    url = f"https://twitter-aio.p.rapidapi.com/search/{searchterm}"
    querystring = {}
    if category:
        querystring['category'] = category
    if cursor:
        querystring['cursor'] = cursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-aio.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

