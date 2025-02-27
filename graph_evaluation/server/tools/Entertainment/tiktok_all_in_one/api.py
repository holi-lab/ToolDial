import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def feed_trendings(region: str='US', device_id: int=7523368224928586752, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can GET Feed(Trending) videos like you get in first screen of the app for any region(country) you want."
    region: Specify a region for Feed(Trending) videos.
Supported any region like: RU, US, GB, DE, FR, ES, AU, BR, CA, GR, IL, IT, MX, NL, PE, PL, SE, UA, VE, etc...
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/feed"
    querystring = {}
    if region:
        querystring['region'] = region
    if device_id:
        querystring['device_id'] = device_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get general info about user by id"
    
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/user"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search users by user name"
    
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/search/user"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_recommendation(region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of popular users in your region/country."
    region: Specify a region to get list of relevant users from.
Supported any region like: RU, US, GB, DE, FR, ES, AU, BR, CA, GR, IL, IT, MX, NL, PE, PL, SE, UA, VE, etc...
Default is US
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/user/recommend"
    querystring = {}
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_suggested(is_id: int, cursor: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion list of users that similar to another.
		This is "Suggested" tab in user profile."
    id: User id to get list of similar users
        cursor: \"cursor\" field from previous request response to get next list of users(offset for pagination)
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/user/suggest"
    querystring = {'id': is_id, }
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers(is_id: int, min_time: int=1640026682, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get users followers list.
		This is "Followers" button in user profile."
    id: User id for what you want get followers
        min_time: \\\"min_time\\\" field from previous request response, is used to get next list of users(pagination)
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/user/follower"
    querystring = {'id': is_id, }
    if min_time:
        querystring['min_time'] = min_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following(is_id: int, min_time: int=1640026682, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user following list.
		This is "Following" button in user profile."
    id: User id of the user you want get following list
        min_time: \"min_time\" field from previous request response, is used to get next list of users(pagination)
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/user/following"
    querystring = {'id': is_id, }
    if min_time:
        querystring['min_time'] = min_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discover(offset: int=20, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of trending videos with specific hashtag for region(country).
		This is "Discover" screen in the mobile app."
    offset: Offset to get more results if \\\\\\\"has_more\\\\\\\" field equal 1
It can be any positive integer or you can get it from \\\\\\\"cursor\\\\\\\" field
        region: Specify a region(default US).
Supported any region like: RU, US, GB, DE, FR, ES, AU, BR, CA, GR, IL, IT, MX, NL, PE, PL, SE, UA, VE, etc...
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/discover"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_videos(is_id: int, offset: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all videos that contain specific hashtag(challenge)"
    id: Hashtag(challenge) id you can find in"cid" field (for example in hashtag search response)
        offset: Offset to get more results if "has_more" field equal 1
It can be any positive integer or you can get it from "cursor" field
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/hashtag/videos"
    querystring = {'id': is_id, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_challenge_info(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get general info about hashtag(challenge)"
    id: Hashtag(challenge) id you can find in"cid" field (for example in hashtag search response)
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/hashtag"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos_for_music(is_id: int, offset: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info about all videos that use specific music(sound)"
    id: Music id you can find in\"mid\" field (for example in music search response)
        offset: Offset to get more results if \"has_more\" field equal 1
It can be any positive integer or you can get it from \"cursor\" field
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/music/video"
    querystring = {'id': is_id, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_info(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all info about specific music(sound)."
    id: Music id you can find in"mid" field (for example in music search response)
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/music"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_videos(is_id: int, max_cursor: int=1632138733000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all videos that specific user uploaded."
    id: User id you can find in \\\"uid\\\" field
        max_cursor: Offset to get more results if in a response you get field \\\"has_more\\\" equal to 1 then you can use \\\"max_cursor\\\" field for this parameter
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/user/videos"
    querystring = {'id': is_id, }
    if max_cursor:
        querystring['max_cursor'] = max_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_search(query: str, region: str='US', offset: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ongoing lives by query string.
		This is "Live" tab in the mobile app search page.
		You can find direct video url in result."
    query: Keyword you want to find in live streams
        region: Specify a region(default US).
Supported any region like: RU, US, GB, DE, FR, ES, AU, BR, CA, GR, IL, IT, MX, NL, PE, PL, SE, UA, VE, etc...
        offset: Offset to get more results if \\\"has_more\\\" field equal 1
It can be any positive integer or you can get it from \\\"cursor\\\" field
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/search/live"
    querystring = {'query': query, }
    if region:
        querystring['region'] = region
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_challenge_search(query: str, offset: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtags(challenges) by query string.
		This is "Hashtags" tab in the mobile app search page"
    query: Keyword you want to find in hashtags(challenges)
        offset: Offset to get more results if \"has_more\" field equal 1
It can be any positive integer or you can get it from \"cursor\" field
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/search/hashtag"
    querystring = {'query': query, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_suggestion(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion for search query.
		Suggestion from TikTok for search."
    query: Keyword you need to get suggestions for.
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/search/suggest"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def music_search(query: str, sort: int=2, offset: int=20, filter: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get music by query string.
		This is "Sounds" tab in the mobile app search page"
    query: Keyword you want to find in music
        sort: Sort result.
Possible values(default 0):
0 - relevance;
1 - most used;
2 - most recent;
3 - shortest;
4 - longest;
        offset: Offset to get more results if \"has_more\" field equal 1
It can be any positive integer or you can get it from \"cursor\" field
        filter: Filter result.
Possible values(default 0):
0 -all
1 - title
2 - creators
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/search/music"
    querystring = {'query': query, }
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videos_search(query: str, offset: int=20, sort: int=1, time: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos by query string.
		This is "Video" tab in the mobile app search page"
    query: Keyword you want to find in videos
        offset: Offset to get more results if \\\\\\\\\\\\\\\"has_more\\\\\\\\\\\\\\\" field equal 1
It can be any positive integer or you can get it from \\\\\\\\\\\\\\\"cursor\\\\\\\\\\\\\\\" field
        sort: Sorting you want apply to result.
Possible values(default 0):
0 - relevance; 
1 - most liked
        time: Filter by date you want apply to result.
Possible values(default 0):
0 - all time; 
1 - yesterday; 
7 - this week;
30 - month;
90 - 3 month;
180 - 6 month
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/search/video"
    querystring = {'query': query, }
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top(offset: int=20, sort: str=None, query: str='funny cats', time: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get everything that related to your query: users, videos, music, hashtags, etc.
		It is first tab in the app if you hit search button."
    offset: Offset to get more results if \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"has_more\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" field equal 1
It can be any positive integer or you can get it from \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"cursor\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" field
        sort: Sorting you want apply to result.
Possible values(default 0):
0 - relevance; 
1 - most liked
        query: This is what you want to find
        time: Filter by date you want apply to result.
Possible values(default 0):
0 - all time; 
1 - yesterday; 
7 - this week;
30 - month;
90 - 3 month;
180 - 6 month
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/search/top"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if query:
        querystring['query'] = query
    if time:
        querystring['time'] = time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_reply(video_id: int, comment_id: int, offset: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reply comments to specific comment for video"
    video_id: Id of the video that owns the comment
        comment_id: Comment id that you can get from comments endpoint from \\\\\\\"cid\\\\\\\" field
        offset: Offset to get more comments. You can pass \\\\\\\"cursor\\\\\\\" field from previous request if \\\\\\\"has_more\\\\\\\" field equal 1
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/video/comments/reply"
    querystring = {'video_id': video_id, 'comment_id': comment_id, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_comments(is_id: int, offset: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all comments for specific video"
    id: Video id you can find it in \\\\\\\"aweme_id\\\\\\\" field
        offset: Offset to get more comments if \\\\\\\"has_more\\\\\\\" field equal 1
It can be any positive integer or you can get it from \\\\\\\"cursor\\\\\\\" field
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/video/comments"
    querystring = {'id': is_id, }
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_info(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all info about video by id"
    id: Retrieve this value from field \\\"aweme_id\\\" or from link for example:
link: https://www.tiktok.com/@watchs01/video/7013267270252252421
video id(aweme_id): 7013267270252252421
        
    """
    url = f"https://tiktok-all-in-one.p.rapidapi.com/video"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tiktok-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

