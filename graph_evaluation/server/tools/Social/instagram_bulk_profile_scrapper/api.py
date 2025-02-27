import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def music_search_by_keyword(query: str, nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Contact me for Custom package or requirements**
		
		Search Music by Keyword
		
		You can pass nextMaxId to fetch more reels"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/audio"
    querystring = {'query': query, }
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulk_profile_fast_response(ig: str, response_type: str, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Contact me for Custom package or requirements**
		
		Fetch Instagram Short Profile without recent feeds, Useful for Profile verification, This endpoint use smart caching algorithm, Contact me to reduce caching time."
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'ig': ig, 'response_type': response_type, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulk_profile_by_pk(ig: str, response_type: str, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API return time will be faster, As this fetch  profile data directly from userid (pk)
		
		Scrap instagram profile by pk"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'ig': ig, 'response_type': response_type, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reels_by_music_id(request_type: str, query: str, nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instagram reels music by id
		
		You can use nextMaxId to fetch next page"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/music"
    querystring = {'request_type': request_type, 'query': query, }
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_feed_by_location_id(loc_id: str, feed_type: str, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram Feeds by Location"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/media_by_loc"
    querystring = {'loc_id': loc_id, 'feed_type': feed_type, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_feed_by_username(ig: str, response_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram Profile Feeds by Username"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'ig': ig, 'response_type': response_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_feed_by_hashtags(tag: str, feed_type: str, nextmaxid: str=None, corsenabled: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram Feeds by Hashtag"
    nextmaxid: Use to load next batch. Pass this value from  previous api response
        
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/media_by_tag"
    querystring = {'tag': tag, 'feed_type': feed_type, }
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_username(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Contact me for Custom package or requirements**
		
		This API will be usefull for Instagram username validation or finding similar usernames to pick correct one."
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/username_validation"
    querystring = {'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pk_to_username(pk: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Username from pk/id"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/pk_to_username"
    querystring = {'pk': pk, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advance_bulk_profile(response_type: str, ig: str, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrap email and phone number using this api
		
		Scrap Instagram Profile using username.
		
		This API response take around 5-6 seconds scrap one profile with post.
		
		It is designed and capable to handle bulk request. but we recommend to add 5-10 seconds random delay between each request.
		
		
		This API also support optional parameter corsEnabled to generate image/video link which can be directly used in Browser"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/advance_profile"
    querystring = {'response_type': response_type, 'ig': ig, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reels_by_pk(response_type: str, ig: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Instagram reels/clips from username"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'response_type': response_type, 'ig': ig, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reels_with_cursor(ig: str, nextmaxid: str=None, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrap instagram reels by pk or username"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/reels"
    querystring = {'ig': ig, }
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_search(name: str, corsenabled: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Location search on Instagram"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/search_locations"
    querystring = {'name': name, 'corsEnabled': corsenabled, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtags_search(name: str, corsenabled: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search hashtags on Instagram"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/search_hashtags"
    querystring = {'name': name, 'corsEnabled': corsenabled, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tagged_posts(corsenabled: str, ig: str, nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instagram Users Tagged posts"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/tagged"
    querystring = {'corsEnabled': corsenabled, 'ig': ig, }
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suggested_profiles(response_type: str, ig: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instagram Discover Profiles endpoint"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'response_type': response_type, 'ig': ig, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def story_highlights(ig: str, response_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instagram Story Highlights"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'ig': ig, 'response_type': response_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers_by_username(username: str, corsenabled: str='false', nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Contact me for Custom package or requirements**
		
		Fetch followers list
		
		Carry forward nextMaxId to retrieve next batch"
    nextmaxid: Leave this empty in first batch
        
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/followers"
    querystring = {'username': username, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post_by_shortcode(shortcode: str, response_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Contact me for Custom package or requirements**
		
		Fetch Instagram post/feed from shortcode"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/media_by_id"
    querystring = {'shortcode': shortcode, 'response_type': response_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_username(ig: str, response_type: str, corsenabled: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Contact me for Custom package or requirements**
		
		This API will be usefull for Instagram username validation or finding similar usernames to pick correct one.
		
		This API also support CORS enabled image URL."
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'ig': ig, 'response_type': response_type, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulk_profile(ig: str, response_type: str, corsenabled: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrap Instagram Profile using username.
		
		This API response take around 5-6 seconds scrap one profile with post.
		
		It is designed and capable to handle bulk request. but we recommend to add 5-10 seconds random delay between each request.
		
		
		This API also support optional parameter corsEnabled to generate image/video link which can be directly used in Browser"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'ig': ig, 'response_type': response_type, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulk_profile_cors_enabled(response_type: str, ig: str, corsenabled: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API Images and Video url can be used directly in browser"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'response_type': response_type, 'ig': ig, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followings_by_username(username: str, corsenabled: str='false', nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch following list
		
		Carry forward nextMaxId to retrieve next batch"
    nextmaxid: Leave this empty in first batch
        
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/following"
    querystring = {'username': username, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_story_by_shortcode(response_type: str, shortcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Instagram story from shortcode"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/media_by_id"
    querystring = {'response_type': response_type, 'shortcode': shortcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_story_by_username(response_type: str, ig: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Instagram stories from username"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/ig_profile"
    querystring = {'response_type': response_type, 'ig': ig, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_with_cursor(ig: str, corsenabled: str='false', nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API return time will be faster, As this fetch  profile data directly from userid (pk)
		
		Scrap instagram profile by pk or username"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/feeds"
    querystring = {'ig': ig, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def highlights_tray_by_id(is_id: str, response_type: str, corsenabled: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Instagram story highlights tray list by Highlight id"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/media_by_id"
    querystring = {'id': is_id, 'response_type': response_type, 'corsEnabled': corsenabled, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followers_by_pk(pk: str, nextmaxid: str=None, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch followers list
		
		Carry forward nextMaxId to retrieve next batch"
    nextmaxid: Leave this empty in first batch
        
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/followers"
    querystring = {'pk': pk, }
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def followings_by_pk(pk: str, corsenabled: str='false', nextmaxid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch following list
		
		Carry forward nextMaxId to retrieve next batch"
    nextmaxid: Leave this empty in first batch
        
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/following"
    querystring = {'pk': pk, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    if nextmaxid:
        querystring['nextMaxId'] = nextmaxid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_likers(media_id: str, corsenabled: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram Post Likers list by media_id"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/liker_by_media"
    querystring = {'media_id': media_id, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_commenter(media_id: str, corsenabled: str='false', nextminid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Instagram Post Commenter's list by media_id"
    nextminid: Encoded URL content of whole JSON object 

Browser Console Example:
encodeURIComponent(decodeURI(`Your JSON String`))

Example Value: 
%7B%22cached_comments_cursor%22%3A%20%2218142440796202967%22%2C%20%22bifilter_token%22%3A%20%22KKEBAEJEOfp_qT8AhH3X4HKlPwBEQXS88tI_AIhv3iz-nz8ATKsDM6e7PwANscnnWbE_AM47SjD8fD8AEAQV9lKPQABRH4m_idU_ANP3u_xobj8AWeQ7JEjXQABZUoETn68_ABvmpdJksT8AIkvNnjdoQACoNqIOTKk_AOoixmjiiz8ANJaxT-SlPwD6zgQE9INAADsPnDQIlj8AvF-5NdaZPwAA%22%7D

        
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/commenter_by_media"
    querystring = {'media_id': media_id, }
    if corsenabled:
        querystring['corsEnabled'] = corsenabled
    if nextminid:
        querystring['nextMinId'] = nextminid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_reels_by_shortcode(shortcode: str, response_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Contact me for Custom package or requirements**
		
		Fetch Instagram reels/clips from short code"
    
    """
    url = f"https://instagram-bulk-profile-scrapper.p.rapidapi.com/clients/api/ig/media_by_id"
    querystring = {'shortcode': shortcode, 'response_type': response_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-bulk-profile-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

