import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_child_comments_by_media_id_and_comment_id(media_id: str, comment_id: str, min_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get child comments of post/reel/igtv/etc using media_id and comment_id.
		You can pass `next_min_child_cursor` from response to `min_id` for next child comments.
		
		Please make sure you don't add `/` at the end of url."
    min_id: Leave `min_id` empty for first request then provide `next_min_child_cursor ` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_child_comments/{media_id}/{comment_id}"
    querystring = {}
    if min_id:
        querystring['min_id'] = min_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_insights_of_post_reel_by_shortcode(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hidden insights such as `save_count`, `share_count` of any post, reel using shortcode.  You can find more details [here(tutorial).](https://rapidapi.com/mrngstar/api/instagram-api-20231/tutorials/hidden-insights-of-public-posts,-reels)"
    shortcode: **How to find shortcode?**
www.instagram.com/reel/CrgVBtHr3DP/ here **CrgVBtHr3DP** is shortcode.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_insights_from_shortcode/{shortcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_insights_of_post_reel_by_media_id(media_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hidden insights such as `save_count`, `share_count` of any post, reel using media_id.  You can find more details [here(tutorial).](https://rapidapi.com/mrngstar/api/instagram-api-20231/tutorials/hidden-insights-of-public-posts,-reels)"
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_insights/{media_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_feed_insights(user_id: str, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Feed Insights 
		
		Endpoint is available on request, please contact us for the custom plan.
		**Update:** We are sorry, this endpoint is not available due to some issues from Instagram side, please try media insights from shortcode/media ID endpoint."
    count: Count should be between **1-500**. Default value is 1.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/feed_insights/{user_id}"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_posts(location_id: str, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts from `location_id`. 
		You can use previous endpoint to get location_id from city names. Use `pk` from previous endpoint response as `location_id`"
    location_id: Use previous endpoint to get `location_id` from city names. Use `pk` from previous endpoint response as `location_id` here.
        max_id: Leave `max_id` empty for first request then provide `next_max_id` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/location_media/{location_id}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def server_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check Server Status.
		**This is an optional endpoint, you may ignore it.**"
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_followers_by_user_id(user_id: str, max_id: str=None, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get followers of an Instagram user using user_id."
    max_id: Leave `max_id` empty for first request then provide `next_max_id` received from previous request to get more items.
        count: **Optional**
Count should be between 1-100. If not we will use the default value.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_followers/{user_id}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_posts_by_username(username: str, max_id: str=None, count: int=12, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts of an Instagram user using username.
		*For fast response, we suggest you to use previous endpoint.*"
    max_id: Leave `max_id` empty for first request then provide `next_max_id` received from previous request to get more items.
        count: number of posts
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_posts_from_username/{username}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_giphy_gif(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search gif using string. You can use any language as string. ex. बिल्ली  (in English: cat)"
    string: Use can use any language string to search gif.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/search_giphy/{string}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_facebook_page_facebook_account_connected_to_instagram_account_by_insta_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Facebook page, Facebook user connected to an Instagram account by using user_id.  You can find more details [here](https://rapidapi.com/mrngstar/api/instagram-api-20231/tutorials/get-facebook-page,-facebook-account-details-of-an-instagram-account-1)"
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/fb_details/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info_from_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief Information about an Instagram user using **username**."
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/get_user_info_from_username/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_music_audio(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search music using string. You can use any language as string."
    string: Use can put any language string to search music.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/search_music/{string}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_place_ids_from_city_names(place_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get place ids from place(city) names. You can use these place ids to get posts by city names in next endpoint. 
		You can use any language as place/city name."
    place_name: Use can put place(city) name in any language.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/places/{place_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hashtags(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search hashtags using string. You can use any language as string for ex. 민지."
    string: Use can put any language string to search hashtag.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/search_hashtag/{string}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_posts(hashtag: str, feed_type: str, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts by hashtag. You can use any language as hashtag ex. भारत"
    hashtag: Use can put any language hashtag. 
*If you want to search hashtags for given string then please use previous endpoint.*
        feed_type: feed_type should be **recent** or **top**
        max_id: Leave `max_id` empty for first request then provide `next_max_id` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/hashtag_media/{hashtag}"
    querystring = {'feed_type': feed_type, }
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_id_pk_to_username_fast_response(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch username & other basic details by using user_id(pk)."
    user_id: Instagram user_id(pk)
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/get_username/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_tags_posts_by_user_id(user_id: str, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user tag posts of an Instagram user using user_id.
		User tag posts can be found at www. instagram .com/{username}/tagged/"
    max_id: Leave `max_id` empty for first request then provide `next_max_id` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_tags/{user_id}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hashtags_followed_by_an_user_using_user_id(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hashtags followed by an Instagram user using user_id."
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/followed_hashtags/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_info_from_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief Information about an Instagram user using user_id"
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/get_user_info/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def username_to_user_id(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch User Id & other details by using Instagram username"
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/get_user_id/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_story_post_reel_other_info_by_media_id(media_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief media (story, post, reel & other) info by media_id.
		
		`story_id` can be obtained from url ex. in instagram. com/stories/thegoodquote/3102217023052923777/  here **3102217023052923777** is story_id."
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_info/{media_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_media_of_a_highlight_using_highlight_id(highlight_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all media of a particular  highlight using `highlight_id`. 
		To get all highlight_id's of a user please use previous endpoint."
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/highlight_media/{highlight_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def no_cors_hd_profile_photo(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch no CORS HD profile picture of a Instagram user. 
		No CORS image urls will expire after 30 days from date of creation.
		
		Along with original no CORS image, you will get thumbnail & display_url for the same."
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/nocors_profile/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def no_cors_images_from_shortcode(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch no CORS image urls of posts, stories & reels. Please note currently we only support images. No CORS image urls will expire after 30 days from date of creation.
		**Note: For carousel media having more than 5 images, this endpoint will take some time to generate no CORS image urls for all carousel media of posts.**"
    shortcode: **How to find shortcode?**
In www.instagram.com/p/CsB0gIFAxUB/ , here **CsB0gIFAxUB** is shortcode.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/nocors_media_from_shortcode/{shortcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def no_cors_images_from_media_id(media_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch no CORS image urls of posts, stories & reels. Please note currently we only support images. No CORS image urls will expire after 30 days from date of creation.
		**Note: For carousel media having more than 5 images, this endpoint will take some time to generate no CORS image urls for all carousel media of posts.**"
    media_id: Use post, story, reel media_id
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/nocors_media/{media_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user(string: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search user using string. You can use any language as string for ex. 민지."
    string: Use can put any language string to search user.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/search_user/{string}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_active_archived_stories_of_user_by_date_wise_using_user_id(user_id: int, end_cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get active. archived stories of a user by date wise using user_id.
		Main feature of this endpoint is to get active stories & highlight stories date wise.
		**Note**: archived stories are stories which are present in highlights."
    end_cursor: Leave `end_cursor` empty for first request then provide `end_cursor` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/active_archived_media/{user_id}"
    querystring = {}
    if end_cursor:
        querystring['end_cursor'] = end_cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comments_of_media_by_shortcode(shortcode: str, min_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments of post/reel/igtv/etc using shortcode"
    min_id: Leave `min_id` empty for first request then provide `next_min_id` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_comments_from_shortcode/{shortcode}"
    querystring = {}
    if min_id:
        querystring['min_id'] = min_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comments_by_media_id(media_id: str, min_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments of post/reel/igtv/etc using media_id"
    min_id: Leave `min_id` empty for first request then provide `next_min_id` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_comments/{media_id}"
    querystring = {}
    if min_id:
        querystring['min_id'] = min_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_like_details_of_any_media_by_shortcode(shortcode: str, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all users who liked a particular post/reel/IGTV/etc using shortcode.
		You can fetch upto 9000 users(upto 5MB of response). Please make sure your systems are capable of handling too much data otherwise you can use 1000 count value."
    shortcode: **How to find shortcode?**
www.instagram.com/p/CrgVBtHr3DP/ here **CrgVBtHr3DP** is shortcode.
        count: By default you will get 100 users data. 
1 < count < 9000
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_likes_from_shortcode/{shortcode}"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_like_details_of_any_media_by_media_id(media_id: str, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all users who liked a particular post/reel/IGTV/etc using media_id.
		You can fetch upto 9000 users(upto 5MB of response). Please make sure your systems are capable of handling too much data otherwise you can use 1000 count value."
    count: By default you will get 100 users data. 
1 < count < 9000
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_likes/{media_id}"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_stories_by_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active stories of an Instagram user using username.
		
		***Please use` Get all stories by user_id` endpoint for fast response.***"
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_stories_from_username/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reel_by_media_id(media_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reel info by media_id"
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_info/{media_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reel_by_shortcode(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reel info by shortcode."
    shortcode: **How to find shortcode?**
www.instagram.com/reel/CrgVBtHr3DP/ here **CrgVBtHr3DP** is shortcode.
www.instagram.com/reel/{shortcode}
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_info_from_shortcode/{shortcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_story_post_reel_other_info_by_shortcode(shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief media (story, post, reel & other) info by shortcode."
    shortcode: **How to find shortcode?**
www.instagram.com/p/CrgVBtHr3DP/ here **CrgVBtHr3DP** is shortcode.
www.instagram.com/reel/{shortcode}
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_info_from_shortcode/{shortcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_highlights_tray_by_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user's all highlight `highlight_id` from this endpoint. To get all medias of a particular highlight please use next endpoint."
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_highlight_tray/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_stories_by_user_id(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all active stories of an Instagram user using user_id."
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_stories/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_reels_by_username(username: str, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reels of an Instagram user using username.
		*For fast response, we suggest you to use previous endpoint.*"
    max_id: Leave `max_id` empty for first request then provide `max_id` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_reels_from_username/{username}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_reels_by_user_id(user_id: int, max_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reels of an Instagram user using username."
    max_id: Leave `max_id` empty for first request then provide `max_id` received from previous request to get more items.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_reels/{user_id}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_posts_by_user_id(user_id: str, max_id: str=None, count: int=12, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts of an Instagram user using user_id."
    max_id: Leave `max_id` empty for first request then provide `next_max_id` received from previous request to get more items.
        count: number of posts
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_posts/{user_id}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_followings_by_user_id(user_id: int, max_id: str=None, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get followings of an Instagram user using user_id."
    max_id: Leave `max_id` empty for first request then provide `next_max_id` received from previous request to get more items.
        count: **Optional**
Count should be between 1-100. If not we will use the default value.
        
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/user_following/{user_id}"
    querystring = {}
    if max_id:
        querystring['max_id'] = max_id
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_public_contact_details_by_user_id(user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get public contact details such as phone number, email, address, etc of an Instagram user using user_id"
    
    """
    url = f"https://instagram-api-20231.p.rapidapi.com/api/public_contact_info/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

