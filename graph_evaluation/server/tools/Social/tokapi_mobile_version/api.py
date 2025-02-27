import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_search_location(count: int=10, offset: int=0, keyword: str='London', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search locations by text query"
    offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        keyword: Query text
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/search/location"
    querystring = {}
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_location_posts_location_id(location_id: str, offset: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get videos by location ID"
    offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/location/posts/{location_id}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_location_location_id(location_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location info by ID"
    
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/location/{location_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_user_id_followings(user_id: str, count: int=10, offset: int=0, ids_only: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User following list"
    user_id: user's ID can be uid or sec_uid
        offset: If in a response you get parameter has_more equal to 1 then you also have min_time value for a next request
        ids_only: If 1 will return only followings IDs, otherwise will return full information
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/{user_id}/followings"
    querystring = {}
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if ids_only:
        querystring['ids_only'] = ids_only
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_user_id_followers(user_id: str, offset: int=0, count: int=10, ids_only: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User followers"
    user_id: user's ID can be uid or sec_uid
        offset: If in a response you get parameter has_more equal to 1 then you also have min_time value for a next request
        ids_only: If 1 will return only followers IDs, otherwise will return full information
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/{user_id}/followers"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if ids_only:
        querystring['ids_only'] = ids_only
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_user_id_qr_code(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "QR code by user ID"
    user_id: user's ID can be uid or sec_uid
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/{user_id}/qr_code"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_user_id(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User information by ID"
    user_id: user's ID can be uid or sec_uid
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_username_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user identifiers by username"
    username: username
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/username/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_user_id_playlist_playlist_id_videos(user_id: str, playlist_id: str, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Videos by playlist ID"
    user_id: user's ID can be uid or sec_uid
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/{user_id}/playlist/{playlist_id}/videos"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_user_id_playlist(user_id: str, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User playlists"
    user_id: user's ID can be uid or sec_uid
        offset: If in a response you get parameter has_more equal to 1 then you also have min_time value for a next request
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/{user_id}/playlist"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_user_id_playlist_playlist_id(user_id: str, playlist_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Playlist information by ID"
    user_id: user's ID can be uid or sec_uid
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/{user_id}/playlist/{playlist_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_user_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User information by username"
    
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/user/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_post_user_user_id_posts(user_id: str, offset: int=0, count: int=10, with_pinned_posts: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User video feed"
    user_id: user's ID can be uid or sec_uid
        offset: If in a response you get parameter hasMore equal to 1 then you also have max_cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/post/user/{user_id}/posts"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if with_pinned_posts:
        querystring['with_pinned_posts'] = with_pinned_posts
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_search_live(keyword: str, offset: int=0, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search live streams by text query"
    keyword: Query text
        offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/search/live"
    querystring = {'keyword': keyword, }
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_post_user_user_id_liked_posts(user_id: str, count: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User liked video feed"
    user_id: user's ID can be uid or sec_uid
        offset: If in a response you get parameter hasMore equal to 1 then you also have max_cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/post/user/{user_id}/liked_posts"
    querystring = {}
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_search_music(keyword: str, count: int=10, offset: int=0, filter_by: str=None, sort_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search music by text query"
    keyword: Query text
        offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        filter_by: Filter by type, can be empty or one of: `0` - All, `1` - Title, `2` - Creators
        sort_type: Sort type, can be empty or one of: `0` - Relevance, `1` - Most used, `2` - Most recent, `3` - Shortest, `4` - Longest
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/search/music"
    querystring = {'keyword': keyword, }
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if filter_by:
        querystring['filter_by'] = filter_by
    if sort_type:
        querystring['sort_type'] = sort_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_search_user(keyword: str, count: int=10, follower_count: str=None, profile_type: str=None, other_pref: str=None, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search users by text query"
    follower_count: Filter by followers count, can be empty or one of: `ZERO_TO_ONE_K`(0 to 1k), `ONE_K_TO_TEN_K`(1k to 10k), `TEN_K_TO_ONE_H_K`(10k to 100k), `ONE_H_K_PLUS`(100k+)
        profile_type: Filter by user profile type, can be empty or `VERIFIED`
        other_pref: Filter by other preference, can be empty or one of: `USERNAME` (keyword usage in username)
        offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/search/user"
    querystring = {'keyword': keyword, }
    if count:
        querystring['count'] = count
    if follower_count:
        querystring['follower_count'] = follower_count
    if profile_type:
        querystring['profile_type'] = profile_type
    if other_pref:
        querystring['other_pref'] = other_pref
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_search_post(keyword: str, offset: int=0, sort_type: str=None, publish_time: str=None, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos by text query"
    keyword: Query text
        offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        sort_type: Sort type, can be empty or one of: `1` - Most liked, `0` - Relevance
        publish_time: Date posted filter, can be empty or one of: `0` - All time, `1` - Yesterday, `7` - This week, `30` - This month, `90` - Last 3 months, `180` - Last 6 months
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/search/post"
    querystring = {'keyword': keyword, }
    if offset:
        querystring['offset'] = offset
    if sort_type:
        querystring['sort_type'] = sort_type
    if publish_time:
        querystring['publish_time'] = publish_time
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_post(video_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video by web URL, can be vm.tiktok.com or tiktok.com"
    
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/post"
    querystring = {'video_url': video_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_post_post_id(post_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video by ID"
    
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/post/{post_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_search_hashtag(keyword: str, count: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search hashtags by text query"
    keyword: Query text
        offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/search/hashtag"
    querystring = {'keyword': keyword, }
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_post_post_id_comments(post_id: str, count: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Comments by video ID"
    offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/post/{post_id}/comments"
    querystring = {}
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_category(count: int=10, offset: int=0, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trending categories list"
    offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/category"
    querystring = {}
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_feed_recommended(pull_type: int=0, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trending feed"
    pull_type: The type of loading feed (0 - Initial loading, 2 - Load more, 8 - Reload)
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/feed/recommended"
    querystring = {}
    if pull_type:
        querystring['pull_type'] = pull_type
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_music_music_id(music_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Music information by ID"
    
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/music/{music_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_music_posts_music_id(music_id: str, offset: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos by music ID"
    offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/music/posts/{music_id}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_sticker(sticker_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Effects information by IDs (look at stickers field in video response)"
    sticker_ids: Comma separated ids
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/sticker"
    querystring = {'sticker_ids': sticker_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_hashtag_posts_hashtag_id(hashtag_id: str, count: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos by hashtag ID"
    offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/hashtag/posts/{hashtag_id}"
    querystring = {}
    if count:
        querystring['count'] = count
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_hashtag_hashtag_id(hashtag_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search videos by hashtag ID"
    
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/hashtag/{hashtag_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_sticker_sticker_id(sticker_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Effect information by ID (look at stickers field in video response)"
    
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/sticker/{sticker_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_sticker_posts_sticker_id(sticker_id: str, offset: int=0, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Video by effect ID (look at stickers field in video response)"
    offset: If in a response you get parameter hasMore equal to 1 then you also have cursor value for a next set
        
    """
    url = f"https://tokapi-mobile-version.p.rapidapi.com/v1/sticker/posts/{sticker_id}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokapi-mobile-version.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

