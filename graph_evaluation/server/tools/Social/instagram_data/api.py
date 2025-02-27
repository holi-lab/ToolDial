import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def location_stories(location_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location user stories by using set of location ids values.  
		
		**Location ids can be found by using /location/search endpoint**"
    location_ids: Location ids should specified in the following format: **LOCATION_ID,LOCATION_ID,...**

Location ids can be found by using **/location/search** endpoint

For example: 
2082352,75929182,6811413,130521400908152,59736,1161978,221177873
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/location/stories"
    querystring = {'location_ids': location_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_search(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get locations metadata (id and etc)"
    location: Location name, for example: new york
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/location/search"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_reels_feed(hashtag: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag reels feed"
    hashtag: Instagram hashtag

Values accepted:
 - summer
        end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/hashtag/feed/reels"
    querystring = {'hashtag': hashtag, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_guides_feed(max_id: str=None, username: str='instagram', user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user guides feed
		
		**NOTE:** Profile should be public(not private)
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster"
    max_id: Pagination cursor. 
To get next batch of data, paste here **next_max_id** value that you have received in previous request response.
        username: Instagram username
        user_id: By using **user_id** instead of the username your request will be executed much faster
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/guides/feed"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_tagged_feed(user_id: str=None, end_cursor: str=None, username: str='instagram', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user tagged feed, post where user was tagged
		
		**NOTE:** Profile should be public(not private)
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster"
    user_id: By using **user_id** instead of the username your request will be executed much faster
        end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        username: Instagram username
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/tagged/feed"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_comments_v2(post: str, next_min_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post comments V2 - more accurate data
		
		**NOTE:** Profile should be public(not private)"
    post: Post example url:
- https://www.instagram.com/p/CAVeEm1gDh2/
        next_min_id: Pagination cursor. 
To get next batch of data, paste here **next_min_id** value that you have received in previous request response.
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/comments/v2"
    querystring = {'post': post, }
    if next_min_id:
        querystring['next_min_id'] = next_min_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_information_metadata(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user information(followers, followings and etc)"
    username: Instagram username
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/info"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_contact_details_email_phone_and_etc(username: str='bibiana_magaji', user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user contact details such as email, phone and etc
		
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster
		**NOTE: the output will include email, phone only IF THESE DATA IS AVAILABLE in IG**"
    username: Instagram username
        user_id: By using **user_id** instead of the username your request will be executed much faster
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/contacts"
    querystring = {}
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_info(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag metadata, top post, total posts and etc"
    hashtag: Instagram hashtag

Values accepted:
 - summer
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/hashtag/info"
    querystring = {'hashtag': hashtag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_feed_v2(hashtag: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag post feed V2"
    hashtag: Instagram hashtag

Values accepted:
 - summer
        end_cursor: Pagination cursor. 
To get next batch of data, paste here **next_max_id** value that you have received in previous request response.
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/hashtag/feed/v2"
    querystring = {'hashtag': hashtag, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_feed_v2(username: str, end_cursor: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user post feed v2
		
		**NOTE:** Profile should be public(not private)"
    username: Instagram username. 

Values accepted:
 - instagram
 - https://www.instagram.com/instagram/
        end_cursor: Pagination cursor. 
To get next batch of data, paste here ** next_max_id** value that you have received in previous request response.
        limit: Limit number of posts to output. 
Min 1 
Max 30
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/feed/v2"
    querystring = {'username': username, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_highlight_reels_feed(user_id: str=None, username: str='dhairiusnyc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user highlight reels feed
		
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster
		**NOTE:** Profile should be public(not private)"
    user_id: By using **user_id** instead of the username your request will be executed much faster
        username: Username

For example: dhairiusnyc
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/highlight/reels"
    querystring = {}
    if user_id:
        querystring['user_id'] = user_id
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_stories(username: str='ghazalmia', user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve active user stories
		
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster"
    username: Instagram username
        user_id: By using **user_id** instead of the username your request will be executed much faster
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/stories"
    querystring = {}
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers(username: str='instagram', end_cursor: str=None, user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followers
		
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster
		**NOTE:** Profile should be public(not private)"
    username: Instagram username. 

Values accepted:
 - instagram
 - https://www.instagram.com/instagram/
        end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        user_id: By using **user_id** instead of the username your request will be executed much faster
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/followers"
    querystring = {}
    if username:
        querystring['username'] = username
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_feed(end_cursor: str=None, username: str='instagram', limit: int=None, user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user post feed
		
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster
		**NOTE:** Profile should be public(not private)"
    end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        username: Instagram username. 

Values accepted:
 - instagram
 - https://www.instagram.com/instagram/
        limit: Limit number of posts to output. 
Min 1 
Max 50
        user_id: By using **user_id** instead of the username your request will be executed much faster
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/feed"
    querystring = {}
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    if username:
        querystring['username'] = username
    if limit:
        querystring['limit'] = limit
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def audio_feed(audio_id: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get audio post feed"
    audio_id: Audio id

For example: 
https://www.instagram.com/reels/audio/921447351682109/

**921447351682109 - will be the audio_id**

        end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/audio/feed"
    querystring = {'audio_id': audio_id, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_feed(hashtag: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtag post feed"
    hashtag: Instagram hashtag

Values accepted:
 - summer
        end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/hashtag/feed"
    querystring = {'hashtag': hashtag, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_story_highlight_metadata(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user story highlight metadata from a direct url to a story"
    url: Direct url to a user story highlight

Example: https://www.instagram.com/stories/highlights/17866745050538306/
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/stories/highlights"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_reels_feed(end_cursor: str=None, limit: int=25, user_id: str=None, username: str='instagram', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user reels feed
		
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster"
    end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        limit: Number of items(posts) to return

- Default value: 25
- Limit: 150
        user_id: By using **user_id** instead of the username your request will be executed much faster
        username: Instagram username
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/reels"
    querystring = {}
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    if limit:
        querystring['limit'] = limit
    if user_id:
        querystring['user_id'] = user_id
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_information_metadata_by_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user information(followers, followings and etc) by using user id (numbers)"
    user_id: Instagram username
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/info/id"
    querystring = {'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followings(end_cursor: str=None, username: str='instagram', user_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user followings
		
		**NOTE:** By using **user_id** instead of the username your request will be executed much faster
		**NOTE:** Profile should be public(not private)"
    end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        username: Instagram username. 

Values accepted:
 - instagram
 - https://www.instagram.com/instagram/
        user_id: By using **user_id** instead of the username your request will be executed much faster
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/followings"
    querystring = {}
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    if username:
        querystring['username'] = username
    if user_id:
        querystring['user_id'] = user_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_metadata(post: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post metadata
		
		**NOTE:** Profile should be public(not private)"
    post: Instagram post url. 

Two url formats are accepted:

-   https://www.instagram.com/p/CG5a3RcDb8X/
-   CG5a3RcDb8X
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/post/info"
    querystring = {'post': post, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_feed(location_id: int, end_cursor: str='2312589909032003834', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location post feed"
    location_id: Location ID

For example in this link **https://www.instagram.com/explore/locations/213385402/london-united-kingdom/** the location_id value will be **213385402**

Values accepted:
 - 213385402
        end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/location/feed"
    querystring = {'location_id': location_id, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_comments(post: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post comments
		
		**NOTE:** Profile should be public(not private)"
    post: Post

Values accepted:
- CAVeEm1gDh2
- https://www.instagram.com/p/CAVeEm1gDh2/
        end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/comments"
    querystring = {'post': post, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_likers(post: str, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get users that liked specific post
		
		**NOTE:** Profile should be public(not private)"
    post: Post

Values accepted:
- CAVeEm1gDh2
- https://www.instagram.com/p/CAVeEm1gDh2/
        end_cursor: Pagination cursor. 
To get next batch of data, paste here **end_cursor** value that you have received in previous request response.
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/likers"
    querystring = {'post': post, }
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_search(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a users by using keyword"
    keyword: Any keyword
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/user/search"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_search(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a hashtags by using keyword"
    keyword: Any keyword
        
    """
    url = f"https://instagram-data1.p.rapidapi.com/hashtag/search"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

