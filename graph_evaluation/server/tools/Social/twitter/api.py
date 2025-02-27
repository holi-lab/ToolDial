import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_v2_with_custom_query(query: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://twitter32.p.rapidapi.com/getSearchV2WithCustomQuery"
    querystring = {'query': query, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweets_by_username(username: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this returns tweets based on twitter username.
		you can use cursor input for pagination."
    
    """
    url = f"https://twitter32.p.rapidapi.com/getTweetsByUsername"
    querystring = {'username': username, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_by_user_id(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Profile by user id"
    
    """
    url = f"https://twitter32.p.rapidapi.com/getProfileByUserId"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(start_date: str=None, username: str=None, end_date: str=None, min_likes: str=None, min_retweets: str=None, to_username: str=None, min_replies: str=None, cursor: str=None, lang: str='en', replies: str=None, mention_username: str=None, retweets: str=None, hashtag: str='nike', exact_phrase: str=None, links: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    start_date: example: 2020-01-01
        username: example: nike (without @)
        end_date: example: 2022-01-01
        min_likes: example: 280
tweets with at least 280 likes
        min_retweets: example: 280
tweets with at least 280 retweets
        to_username: example: nike (without @)
tweets sent in reply to nike
        min_replies: example: 280
tweets with at least 280 replies
        replies: you can just send 0 or 1
0 : without reply tweets
1 : only reply tweets
        mention_username: example: nike (without @)
tweets mentions nike
        retweets: you can just send 0 or 1
0 : without retweets
1 : only retweets
        hashtag: example: nike (without #)
        exact_phrase: example: happy hour
contains the exact phrase “happy hour”
        links: you can just send 0 or 1
0 : without link tweets
1 : only link tweets
        
    """
    url = f"https://twitter32.p.rapidapi.com/getSearch"
    querystring = {}
    if start_date:
        querystring['start_date'] = start_date
    if username:
        querystring['username'] = username
    if end_date:
        querystring['end_date'] = end_date
    if min_likes:
        querystring['min_likes'] = min_likes
    if min_retweets:
        querystring['min_retweets'] = min_retweets
    if to_username:
        querystring['to_username'] = to_username
    if min_replies:
        querystring['min_replies'] = min_replies
    if cursor:
        querystring['cursor'] = cursor
    if lang:
        querystring['lang'] = lang
    if replies:
        querystring['replies'] = replies
    if mention_username:
        querystring['mention_username'] = mention_username
    if retweets:
        querystring['retweets'] = retweets
    if hashtag:
        querystring['hashtag'] = hashtag
    if exact_phrase:
        querystring['exact_phrase'] = exact_phrase
    if links:
        querystring['links'] = links
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_v2(min_retweets: str=None, cursor: str=None, mention_username: str=None, retweets: str=None, exact_phrase: str=None, replies: str=None, end_date: str=None, min_replies: str=None, min_likes: str=None, lang: str='en', links: str=None, username: str=None, to_username: str=None, hashtag: str='nike', start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    min_retweets: example: 280
tweets with at least 280 retweets
        mention_username: example: nike (without @)
tweets mentions nike
        retweets: you can just send 0 or 1
0 : without retweets
1 : only retweets
        exact_phrase: example: happy hour
contains the exact phrase “happy hour”
        replies: you can just send 0 or 1
0 : without reply tweets
1 : only reply tweets
        end_date: example: 2022-01-01
        min_replies: example: 280
tweets with at least 280 replies
        min_likes: example: 280
tweets with at least 280 likes
        links: you can just send 0 or 1
0 : without link tweets
1 : only link tweets
        username: example: nike (without @)
        to_username: example: nike (without @)
tweets sent in reply to nike
        hashtag: example: nike (without #)
        start_date: example: 2020-01-01
        
    """
    url = f"https://twitter32.p.rapidapi.com/getSearchV2"
    querystring = {}
    if min_retweets:
        querystring['min_retweets'] = min_retweets
    if cursor:
        querystring['cursor'] = cursor
    if mention_username:
        querystring['mention_username'] = mention_username
    if retweets:
        querystring['retweets'] = retweets
    if exact_phrase:
        querystring['exact_phrase'] = exact_phrase
    if replies:
        querystring['replies'] = replies
    if end_date:
        querystring['end_date'] = end_date
    if min_replies:
        querystring['min_replies'] = min_replies
    if min_likes:
        querystring['min_likes'] = min_likes
    if lang:
        querystring['lang'] = lang
    if links:
        querystring['links'] = links
    if username:
        querystring['username'] = username
    if to_username:
        querystring['to_username'] = to_username
    if hashtag:
        querystring['hashtag'] = hashtag
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_language_codes_for_search(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint gives a list of language codes required for "getSearch" endpoint"
    
    """
    url = f"https://twitter32.p.rapidapi.com/getLangCode"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_id_by_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert tweet url to id for use in "getTweetById" endpoint"
    
    """
    url = f"https://twitter32.p.rapidapi.com/getTweetIdByUrl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this returns user info by username."
    
    """
    url = f"https://twitter32.p.rapidapi.com/getProfile"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_id_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this returns twitter user id based on twitter username."
    
    """
    url = f"https://twitter32.p.rapidapi.com/getUserId"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweets_by_hashtag(hashtag: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this returns tweets based on twitter hashtag.
		you can use cursor input for pagination."
    
    """
    url = f"https://twitter32.p.rapidapi.com/getTweetsByHashtag"
    querystring = {'hashtag': hashtag, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this returns  auto complete result in twitter search."
    
    """
    url = f"https://twitter32.p.rapidapi.com/getAutoComplete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_likes(user_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Likes"
    
    """
    url = f"https://twitter32.p.rapidapi.com/getUserLikes"
    querystring = {'user_id': user_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_media(user_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Media"
    
    """
    url = f"https://twitter32.p.rapidapi.com/getUserMedia"
    querystring = {'user_id': user_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers(user_id: str, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this returns followers based on twitter user id.
		you can use cursor input for pagination."
    
    """
    url = f"https://twitter32.p.rapidapi.com/getFollowers"
    querystring = {'user_id': user_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweet_by_id(tweet_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "access to one tweet data by its id"
    
    """
    url = f"https://twitter32.p.rapidapi.com/getTweetById"
    querystring = {'tweet_id': tweet_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter32.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

