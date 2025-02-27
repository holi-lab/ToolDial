import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_media_by_location(location_id: int, next_media_ids: str=None, max_id: str=None, next_page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media by location
		Use end_cursor to get the rest of the data"
    max_id: The value of end_cursor for viewing the posts list
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/locations/"
    querystring = {'location_id': location_id, }
    if next_media_ids:
        querystring['next_media_ids'] = next_media_ids
    if max_id:
        querystring['max_id'] = max_id
    if next_page:
        querystring['next_page'] = next_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_posts(id_user: int, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user posts. **Max - 12**
		Use end_cursor to get the rest of the data"
    end_cursor: The value of end_cursor for viewing the posts list
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/posts/"
    querystring = {'id_user': id_user, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_likes_users(shortcode: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram media likes users.  **Max - 12**"
    end_cursor: The value of `end_cursor` for viewing the following list
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/likes/"
    querystring = {'shortcode': shortcode, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_igtv_tv(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram IGTV Media"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/tv_info/"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_more_data(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram Media More Data /p/ - /tv/ - /reel/"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/post_info_v2/"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reel_reel(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram Reel Media"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/reel/"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_p_tv_reel(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram Media /p/ - /tv/ - /reel/"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/post_info/"
    querystring = {'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def no_cors_media_beta(url_media: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get No CORS Media (File storage time from 6 hours to 24)
		
		**Specify a direct link to the media that you received in the response /post_info/, etc.**"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/noCORS/"
    querystring = {'url_media': url_media, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_username_suggestions(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram Username Suggestions
		`Issues free logins available for registration by the specified name and surname`
		
		**With each request, new data**"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/randusername/"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info_by_username(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user info by username"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/info_username/"
    querystring = {'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checking_the_server(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checking the server status"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/server/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_similar_accounts(id_user: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram Similar Accounts"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/similar_accounts/"
    querystring = {'id_user': id_user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_reels_posts_by_username(user: str, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user reels posts by Username. **Max - 10**
		Use end_cursor to get the rest of the data"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/reels_posts_username/"
    querystring = {'user': user, }
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_tv_posts(id_user: int, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user tv posts. **Max - 10**
		Use end_cursor to get the rest of the data"
    end_cursor: The value of end_cursor for viewing the posts list
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/tv_posts/"
    querystring = {'id_user': id_user, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_tv_posts_by_username(user: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user tv posts by Username. **Max - 10**
		Use end_cursor to get the rest of the data"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/tv_posts_username/"
    querystring = {'user': user, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_reels_posts(id_user: int, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user reels posts. **Max - 10**
		Use end_cursor to get the rest of the data"
    max_id: The value of end_cursor for viewing the posts list
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/reels_posts/"
    querystring = {'id_user': id_user, }
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_web_profile_info_by_username(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user Web Profile Info by username"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/web_profile_info/"
    querystring = {'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hashtag(hashtag: str, next_max_id: str=None, next_page: int=None, next_media_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by hashtag on Instagram"
    next_max_id: The value of `next_max_id` for viewing the next hashtag list
        next_page: The value of `next_page `for viewing the next hashtag list for `next_max_id`
        next_media_ids: If the response has `next_media_ids`,  insert it as an array
Example: `[2876468579675572954,  2876432403502018778]`
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/hashtag/"
    querystring = {'hashtag': hashtag, }
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    if next_page:
        querystring['next_page'] = next_page
    if next_media_ids:
        querystring['next_media_ids'] = next_media_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info_by_id_user(id_user: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user info by id_user"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/info/"
    querystring = {'id_user': id_user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_posts_by_username(user: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user posts by Username. **Max - 10**
		Use end_cursor to get the rest of the data"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/posts_username/"
    querystring = {'user': user, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_stories_by_id_strories_hd(id_user: int, id_stories: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stories by id stories and id_user HD Format"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/get_stories_hd/"
    querystring = {'id_user': id_user, 'id_stories': id_stories, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_comments_users(shortcode: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram media comments users. **Max - 12**"
    end_cursor: The value of  `end_cursor ` for viewing the following list

In some cases, `next_min_id ` is returned, you need to enter it in full, example:

` {'cached_comments_cursor': '17963428621651302', 'bifilter_token': 'KDsBEABAACgAGAAYABAACAAIAAgAc-dqvad7v8e-pyIb_1y2EW-KR_EWL_V_MZGYGMAGEALDGMXIBSGBAA='}`
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/comments/"
    querystring = {'shortcode': shortcode, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_stories_hd(id_user: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user stories HD Format"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/stories_hd/"
    querystring = {'id_user': id_user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_stories_by_id_strories(id_stories: int, id_user: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stories by id stories and id_user"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/get_stories/"
    querystring = {'id_stories': id_stories, 'id_user': id_user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_highlights_by_id_highligt(id_highlight: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user highlights  by id_highlight.
		
		**id_highlight** can be obtained from the request to get the "User Highlights tray"
		
		Example:
		
		```
		
		"tray":
		0:
		"id":"highlight:17860963649200530" //<- this
		"latest_reel_media":1635260532
		"seen":NULL
		.......
		```
		
		As a result , id_highlight turns out - **17860963649200530**"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/highlights/"
    querystring = {'id_highlight': id_highlight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_id(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram **user id**"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/user_id/"
    querystring = {'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_followers(id_user: int, next_max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user followers.  **Max - 100**"
    next_max_id: The value of `next_max_id` for viewing the following list
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/followers/"
    querystring = {'id_user': id_user, }
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_following(id_user: int, next_max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user following.  **Max - 100**"
    next_max_id: The value of `next_max_id` for viewing the following list
        
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/following/"
    querystring = {'id_user': id_user, }
    if next_max_id:
        querystring['next_max_id'] = next_max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_info_business_account_email_phone(id_user: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram user info, Business account (email, phone)"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/business/"
    querystring = {'id_user': id_user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search User to Instagram"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/search/"
    querystring = {'user': user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_tagged(id_user: int, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user tagged. **Max 50**"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/tagged/"
    querystring = {'id_user': id_user, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_guides(id_user: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user guides. **Max 50**"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/guides/"
    querystring = {'id_user': id_user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_highlights_tray(id_user: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user highlights tray."
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/highlights_tray/"
    querystring = {'id_user': id_user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_stories(id_user: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instagram user stories"
    
    """
    url = f"https://instagram-scraper-2022.p.rapidapi.com/ig/stories/"
    querystring = {'id_user': id_user, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-scraper-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

