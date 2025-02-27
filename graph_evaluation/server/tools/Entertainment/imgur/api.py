import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def account_album_ids(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all of the ids for all of the albums associated with the account"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/albums/ids"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_album_count(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The number of albums an account contains"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/albums/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_albums(username: str, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the albums associated with the account. Must be logged in as the user to see secret and hidden albums."
    page: integer - allows you to set the page number so you don't have to retrieve all the data at once.
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/albums/{page}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_comments(username: str, sort: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The comments that the user has made"
    sort: 'best', 'worst', 'oldest', or 'newest'. Defaults to 'newest'.
        page: Page number (50 items per page). Defaults to 0.
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/comments/{sort}/{page}"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request standard user information."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_album_information(username: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get More information about a specific album"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/album/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_image_information(username: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return information about a specific image. This endpoint works the same as the Image Endpoint. You can use any of the additional actions that the image endpoint with this endpoint."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/image/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_images(username: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all of the images associated with the account. You can page through the images by setting the page, this defaults to 0."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/images/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_likes(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return an array of images that have been upvoted by the user."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/likes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_images_ids(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of Image IDs that are associated with the account."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/images/ids"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_messages(username: str, new: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all messages sent to the user, formatted as a notification. Required to be logged in to view this information."
    new: false for all notifications, true for only non-viewed notification. Default is true.
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/notifications/messages"
    querystring = {}
    if new:
        querystring['new'] = new
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_settings(user: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the account settings if you're logged in as the user"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{user}/settings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_notifications(username: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all of the notifications associated with the account. Defaults to return only new notifications, however you can set the GET variable, new, to false to return all of the notifications. You're required to be logged in as the user to view this information."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/notifications/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_images_count(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the total number of images associated with the account."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/images/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_statistics(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the statistics about the account."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album_image(album_id: str, image_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about an image in an album, any additional actions found in Image Endpoint will also work."
    album_id: The 5 character string for the album id
        image_id: The id for the image
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/album/{album_id}/image/{image_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get album information"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/album/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment_count(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a count of all of the comments associated with the account."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/comments/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comment information"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/comment/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album_images(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all of the images in the album"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/album/{is_id}/images"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment_ids(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return an array of all of the comment IDs."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/comments/ids"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment_replies(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View the comment and the thread of replies"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/comment/{is_id}/replies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery(showviral: str=None, sort: str=None, page: int=None, section: str=None, window: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the images in the gallery. For example the main gallery is /3/gallery/hot/viral/0"
    showviral: true | false - Show or hide viral images from the 'user' section. Defaults to true
        sort: viral | time - defaults to viral
        page: integer - the data paging number
        section: hot | top | user - defaults to hot
        window: Change the date range of the request if the section is "top", day | week | month | year | all, defaults to day
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/{section}/{sort}/{window}/{page}"
    querystring = {}
    if showviral:
        querystring['showViral'] = showviral
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if section:
        querystring['section'] = section
    if window:
        querystring['window'] = window
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_image(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional information about an image in the gallery"
    id: ID of the gallery image
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/image/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_comment_count(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The number of comments on an item in the gallery"
    id: Gallery item ID
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/{is_id}/comments/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_image_comments(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Comments of a Gallery Image"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/image/{is_id}/comments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_item_votes(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the vote information for an image in the gallery (also works for gallery albums by replacing 'image' with 'album' in the route)"
    id: ID for the gallery item
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/image/{is_id}/votes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_search(q_all: str=None, q_any: str=None, q_exactly: str=None, q_not: str=None, q_type: str=None, q_size_px: str=None, page: int=None, sort: str=None, window: str=None, q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the gallery for a given query string"
    q_all: Search for all of these words (and)
        q_any: Search for any of these words (or)
        q_exactly: Search for exactly this word or phrase
        q_not: Exclude results matching this
        q_type: Show results for any file type, jpg | png | gif | anigif (animated gif) | album
        q_size_px: Size ranges, small (500 pixels square or less) | med (500 to 2,000 pixels square) | big (2,000 to 5,000 pixels square) | lrg (5,000 to 10,000 pixels square) | huge (10,000 square pixels and above)
        page: integer - the data paging number
        sort: time | viral - defaults to time
        window: Change the date range of the request if the sort is "top", day | week | month | year | all, defaults to week
        q: Query string (note: if advanced search parameters are set, this query string is ignored). This parameter also supports boolean operators (AND, OR, NOT) and indices (tag: user: title: ext: subreddit: album: meme:). An example compound query would be 'title: cats AND dogs ext: gif'
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/search/{sort}/{window}/{page}"
    querystring = {}
    if q_all:
        querystring['q_all'] = q_all
    if q_any:
        querystring['q_any'] = q_any
    if q_exactly:
        querystring['q_exactly'] = q_exactly
    if q_not:
        querystring['q_not'] = q_not
    if q_type:
        querystring['q_type'] = q_type
    if q_size_px:
        querystring['q_size_px'] = q_size_px
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if window:
        querystring['window'] = window
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about an image"
    id: ID for the image
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/image/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def notifications(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the notifications for the user that's currently logged in"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/notification"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subreddit_galleries(subreddit: str, sort: str=None, window: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View gallery images for a subreddit"
    subreddit: A valid subreddit name
        sort: time | top - defaults to time
        window: Change the date range of the request if the sort is "top", day | week | month | year | all, defaults to week
        page: integer - the data paging number
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/r/{subreddit}/{sort}/{window}/{page}"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if window:
        querystring['window'] = window
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_comment_ids(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all of the comment ids in an array for a gallery item"
    id: Gallery item ID
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/{is_id}/comments/ids"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_gallery_favorites(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the images the user has favorited in the gallery."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/gallery_favorites"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def notification(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns data about a specific notification"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/notification/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_favorites(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the users favorited images, only accessible if you're logged in as the user."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/favorites"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_gallery_profile(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the totals for the gallery profile."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/gallery_profile"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_submissions(username: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the images a user has submitted to the gallery"
    page: Starting at page 0
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/submissions/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replies(username: str, new: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all of the reply notifications for the user. Required to be logged in as that user"
    new: boolean - false for all notifications, true for only non-viewed notification. Default is true.
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/notifications/replies"
    querystring = {}
    if new:
        querystring['new'] = new
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_gallery(gallery_id: str, page: str, sort: str, window: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View images for a custom gallery"
    gallery_id: The ID for the custom gallery.
        page: integer - the data paging number
        sort: viral | time | top - defaults to viral
        window: Change the date range of the request if the sort is "top", day | week | month | year | all, defaults to week
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/g/{gallery_id}/{sort}/{window}/{page}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_gallery_image(gallery_id: str, image_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View a single image in a custom gallery"
    gallery_id: The ID for the gallery.
        image_id: The ID for the image.
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/g/{gallery_id}/{image_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_user_s_e_mail(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks to see if user has verified their email address"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/account/{username}/verifyemail"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_custom_galleries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a user's custom galleries"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/g"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filtered_out_tags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of the user's filtered out tags"
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/g/filtered_out"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def memes_subgallery_image(image_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View a single image in the memes gallery"
    image_id: The ID for the image.
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/g/memes/{image_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def memes_subgallery(page: str=None, sort: str=None, window: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View images for memes subgallery"
    page: integer - the data paging number
        sort: viral | time | top - defaults to viral
        window: Change the date range of the request if the sort is "top", day | week | month | year | all, defaults to week
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/g/memes/{sort}/{window}/{page}"
    querystring = {}
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if window:
        querystring['window'] = window
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subreddit_image(image_id: str, subreddit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View a single image in the subreddit"
    image_id: The ID for the image.
        subreddit: A valid sub-reddit name
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/r/{subreddit}/{image_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_tag(t_name: str, sort: str=None, page: int=None, window: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View images for a gallery tag"
    t_name: The name of the tag.
        sort: viral | time | top - defaults to viral
        page: integer - the data paging number
        window: Change the date range of the request if the sort is "top", day | week | month | year | all, defaults to week
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/t/{t_name}/{sort}/{window}/{page}"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if window:
        querystring['window'] = window
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_tag_image(image_id: str, t_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View a single image in a gallery tag"
    image_id: The ID for the image.
        t_name: The name of the tag.
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/t/{t_name}/{image_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_item_tags(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View tags for a gallery item"
    id: ID of the gallery item
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/{is_id}/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_gallery_images(page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random set of gallery images."
    page: A page of random gallery images, from 0-50. Pages are regenerated every hour.
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/random/random/{page}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery_album(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional information about an album in the gallery."
    id: ID of the gallery album
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/album/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album_image_comments(is_id: str, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Comment on an image in the gallery."
    sort: best | top | new - defaults to best
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/{is_id}/comments/{sort}"
    querystring = {}
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album_image_comment(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about a specific comment. This action also allows any of the additional actions provided in the Comment Endpoint."
    id: Gallery item ID
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/gallery/{is_id}/comment/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversation_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of all conversations for the logged in user."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/conversations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversation(is_id: str, offset: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a specific conversation. Includes messages."
    offset: Additional offset in current page, defaults to 0.
        page: Page of message thread. Starting at 1 for the most recent 25 messages and counting upwards. Defaults to 1
        
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/conversations/{is_id}/{page}/{offset}"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def default_memes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "et the list of default memes."
    
    """
    url = f"https://imgur-apiv3.p.rapidapi.com/3/memegen/defaults"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imgur-apiv3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

