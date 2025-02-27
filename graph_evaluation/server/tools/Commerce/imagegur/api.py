import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def accountimages2(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To make requests for the current account, you may use `me` as the `{{username}}` parameter. For example, `https://api.imgur.com/3/account/me/images` will request all the images for the account that is currently authenticated."
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/me/images"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountgalleryfavorites3(authorization: str, favoritessort: str, username: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the images the user has favorited in the gallery.
		
		#### Response Model: [Gallery Image](https://api.imgur.com/models/gallery_image) OR [Gallery Album](https://api.imgur.com/models/gallery_album)
		
		
		#### Parameters
		
		| Key  | Required | Description                                                                                     |
		|------|----------|-------------------------------------------------------------------------------------------------|
		| page | optional | integer - allows you to set the page number so you don't have to retrieve all the data at once. |
		| favoriteSort | optional | `oldest`, or `newest`. Defaults to `newest`.                                                    |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/gallery_favorites/{page}/{favoritessort}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def albumcount_un_authed_authed_16(authorization: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the total number of albums associated with the account.
		
		#### Response Model: [Basic](https://api.imgur.com/models/basic)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/albums/count"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountsubmissions5(authorization: str, page: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the images a user has submitted to the gallery. You can add sorting as well after paging. Sorts can be: newest (default), oldest, worst, best. 
		
		#### Response Model: [Gallery Image](https://api.imgur.com/models/gallery_image) OR [Gallery Album](https://api.imgur.com/models/gallery_album)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/submissions/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountavailableavatars_un_authed_authed_6(authorization: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If unauthentiated, get list of default trophies a user can select from. The username need not exist to get the listing.
		
		If authenticated, get the list of available avatars for the given user."
    authorization: If unauthenticated, send the Client-ID
        authorization: If authenticated send the bearer token
        
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/available_avatars"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountbase1(authorization: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request standard user information. If you need the username for the account that is logged in, it is returned in the request for an [access token](/account/current_account). Note: This endpoint also supports the ability to lookup account base info by account ID. To do so, pass the query parameter `account_id`.
		
		#### Response Model: [Account](https://api.imgur.com/models/account)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountavatar_authed_7(authorization: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current account's avatar URL and avatar name."
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/avatar"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commentcount21(authorization: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a count of all of the comments associated with the account.
		
		#### Response Model: [Basic](https://api.imgur.com/models/basic)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/comments/count"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def albumimages1(authorization: str, albumhash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all of the images in the album.
		
		#### Response Model: [Image](https://api.imgur.com/models/image)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/album/{albumhash}/images"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def albums_un_authed_authed_13(authorization: str, username: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the albums associated with the account. Must be logged in as the user to see secret and hidden albums.
		
		#### Response Model: [Album](https://api.imgur.com/models/album)
		
		#### Parameters
		| Key  | Required | Description                                                                                     |
		|------|----------|-------------------------------------------------------------------------------------------------|
		| page | optional | integer - allows you to set the page number so you don't have to retrieve all the data at once. |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/albums/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images23(authorization: str, username: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all of the images associated with the account. You can page through the images by setting the page, this defaults to 0.
		
		#### Response Model: [Image](https://api.imgur.com/models/image)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/images/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def imagecount26(authorization: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the total number of images associated with the account.
		
		#### Response Model: [Basic](https://api.imgur.com/models/basic)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/images/count"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def imageids25(authorization: str, page: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of Image IDs that are associated with the account.
		
		#### Response Model: [Basic](https://api.imgur.com/models/basic)
		"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/images/ids/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image24(authorization: str, imageid: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return information about a specific image. This endpoint works the same as the [Image Endpoint](https://api.imgur.com/endpoints/image/). You can use any of the additional actions that the image endpoint with this endpoint.
		
		#### Response Model: [Image](https://api.imgur.com/models/image)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/image/{imageid}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments18(authorization: str, page: str, commentsort: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the comments the user has created.
		
		#### Response Model: [Comment](https://api.imgur.com/models/comment)
		
		#### Parameters
		
		| Key  | Required | Value                                                         |
		|------|----------|---------------------------------------------------------------|
		| commentSort | optional | `best`, `worst`, `oldest`, or `newest`. Defaults to `newest`. |
		| page | optional | Page number (50 items per page). Defaults to `0`.               |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/comments/{commentsort}/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image0(authorization: str, imagehash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about an image."
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/image/{imagehash}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replies3(authorization: str, commentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the comment with all of the replies for the comment.
		
		#### Response Model: [Comment](https://api.imgur.com/models/comment)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/comment/{commentid}/replies"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallerytag3(authorization: str, tagname: str, window: str, sort: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns tag metadata, and posts tagged with the `tagName` provided"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/t/{tagname}/{sort}/{window}/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallery0(authorization: str, window: str, sort: str, page: str, section: str, mature: str='{{showMature}}', showviral: str='{{showViral}}', album_previews: str='{{albumPreviews}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "| Key       | Required | Value                                                                                             |
		|-----------|----------|---------------------------------------------------------------------------------------------------|
		| section   | optional | `hot` &#124; `top` &#124; `user`. Defaults to `hot`
		| sort      | optional | `viral` &#124; `top` &#124; `time` &#124; `rising` (only available with `user` section). Defaults to `viral` |
		| page      | optional | integer - the data paging number                                                                  |
		| window    | optional | Change the date range of the request if the section is `top`. Accepted values are `day` &#124; `week` &#124; `month` &#124; `year` &#124; `all`. Defaults to `day` |"
    mature: Optional. `true` | `false` - Show or hide mature (nsfw) images in the response section. Defaults to `false` *NOTE:* This parameter is only required if un-authed. The response for authed users will respect their account setting.
        showviral: Optional. `true` | `false` - Show or hide viral images from the `user` section. Defaults to `true`
        album_previews: Optional. `true` | `false` - Include image metadata for gallery posts which are albums 
        
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/{section}/{sort}/{window}/{page}"
    querystring = {'Authorization': authorization, }
    if mature:
        querystring['mature'] = mature
    if showviral:
        querystring['showViral'] = showviral
    if album_previews:
        querystring['album_previews'] = album_previews
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def randomgalleryimages9(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "*DEPRECATED* Returns a random set of gallery images."
    
    """
    url = f"https://imagegur.p.rapidapi.com//"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def albumimage2(authorization: str, albumhash: str, imagehash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about an image in an album, any additional actions found in [Image Endpoint](https://api.imgur.com/endpoints/image/) will also work.
		
		#### Response Model: [Image](https://api.imgur.com/models/image)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/album/{albumhash}/image/{imagehash}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallerysearch8(authorization: str, window: str, page: str, sort: str, q: str='cats', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the gallery with a given query string.
		
		
		#### Parameters
		| Key    | Required | Value                                                                                                        |
		|--------|----------|--------------------------------------------------------------------------------------------------------------|
		| sort   | optional | time &#124; viral &#124; top - defaults to time                                                                        |
		| window | optional | Change the date range of the request if the sort is 'top', day &#124; week &#124; month &#124; year &#124; all, defaults to all. |
		| page   | optional | integer - the data paging number                                                                             |
		
		
		#### Simple Search Query Parameters
		
		| Key | Value                                                                                                                                                                                                                                                                                    |
		|-----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
		| q   | Query string (note: if advanced search parameters are set, this query string is ignored). This parameter also supports boolean operators (AND, OR, NOT) and indices (tag: user: title: ext: subreddit: album: meme:). An example compound query would be 'title: cats AND dogs ext: gif' |
		
		
		
		#### Advanced Search Query Parameters
		
		| Key       | Value                                                                                                                                                                                                |
		|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
		| q_all     | Search for all of these words (and)                                                                                                                                                                  |
		| q_any     | Search for any of these words (or)                                                                                                                                                                   |
		| q_exactly | Search for exactly this word or phrase                                                                                                                                                               |
		| q_not     | Exclude results matching this                                                                                                                                                                        |
		| q_type    | Show results for any file type, jpg  &#124; png  &#124; gif  &#124; anigif (animated gif)  &#124; album                                                                                                                      |
		| q_size_px | Size ranges, small (500 pixels square or less)  &#124; med (500 to 2,000 pixels square)  &#124; big (2,000 to 5,000 pixels square)  &#124; lrg (5,000 to 10,000 pixels square)  &#124; huge (10,000 square pixels and above) |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/search/{sort}/{window}/{page}"
    querystring = {'Authorization': authorization, }
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subredditgalleries1(authorization: str, sort: str, page: str, window: str, subreddit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View gallery images for a subreddit
		
		| Key       | Required | Value                                                                                                        |
		|-----------|----------|--------------------------------------------------------------------------------------------------------------|
		| subreddit | required | pics - A valid subreddit name                                                                                |
		| sort      | optional | `time` &#124; `top` - defaults to time                                                                                |
		| page      | optional | integer - the data paging number                                                                             |
		| window    | optional | Change the date range of the request if the sort is "top". Options are  `day` &#124; `week` &#124; `month` &#124; `year` &#124; `all`. Defaults to week |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/r/{subreddit}/{sort}/{window}/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subredditimage2(authorization: str, subreddit: str, subredditimageid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View a single image in the subreddit
		
		| Key       | Required | Value                         |
		|-----------|----------|-------------------------------|
		| subreddit | required | A valid subreddit name, ie `earthporn` |
		| image_id  | required | The ID for the image.         |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/r/{subreddit}/{subredditimageid}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commentids20(authorization: str, sort: str, page: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return an array of all of the comment IDs.
		
		#### Response Model: [Basic](https://api.imgur.com/models/basic)
		
		#### Parameters
		
		| Key  | Required | Value                                                         |
		|------|----------|---------------------------------------------------------------|
		| commentSort | optional | `best`, `worst`, `oldest`, or `newest`. Defaults to `newest`. |
		| page | optional | Page number (50 items per page). Defaults to `0`.               |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/comments/ids/{sort}/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment19(authorization: str, username: str, commentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return information about a specific comment. This endpoint works the same as the [Comment Endpoint](https://api.imgur.com/endpoint/comment/). You can use any of the additional actions that the comment endpoint allows on this end point."
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/comment/{commentid}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def albumids_un_authed_authed_15(authorization: str, username: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return an array of all of the album IDs (hashes).
		
		#### Response Model: [Basic](https://api.imgur.com/models/basic)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/albums/ids/{page}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verifyuser_se_mail11(authorization: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks to see if user has verified their email address.
		
		#### Response Model: [Basic](https://api.imgur.com/models/basic)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/verifyemail"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallerytags4(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of default tags"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/tags"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album_imagecomments17(authorization: str, galleryhash: str, commentsort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments on an image or album in the gallery.
		
		`galleryHash` is the unique identifier of an album or image in the gallery.
		
		`commentSort` is one of `best` | `top` | `new` - defaults to `best`."
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/{galleryhash}/comments/{commentsort}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def galleryimage13(authorization: str, galleryimagehash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional information about an image in the gallery."
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/image/{galleryimagehash}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album14(authorization: str, albumhash: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional information about an album, this endpoint works the same as the [Album Endpoint](). You can also use any of the additional routes that are used on an album in the album endpoint.
		
		#### Response Model: [Album](https://api.imgur.com/models/album)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/album/{albumhash}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountgalleryprofile10(authorization: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the totals for the gallery profile.
		
		#### Response Model: [Gallery Profile](https://api.imgur.com/models/gallery_profile)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/settings"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountsettings8(authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the account settings, only accessible if you're logged in as the user.
		
		"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/me/settings"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replies28(username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all of the reply notifications for the user. Required to be logged in as that user.
		
		#### Response Model: [Notification](https://api.imgur.com/models/notification)
		
		#### Parameters
		
		| Key | Required | Value                                                                                          |
		|-----|----------|------------------------------------------------------------------------------------------------|
		| new | optional | boolean - `false` for all notifications, `true` for only non-viewed notification. Default is `true`. |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/notifications/replies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comment0(authorization: str, commentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a specific comment.
		
		#### Response Model: [Comment](https://api.imgur.com/models/comment)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/comment/{commentid}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountfavorites4(authorization: str, username: str, page: str, favoritessort: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the users favorited images, only accessible if you're logged in as the user.
		
		#### Response Model: [Gallery Image](https://api.imgur.com/models/gallery_image) OR [Gallery Album](https://api.imgur.com/models/gallery_album)
		
		*Note:* vote data ('ups', 'downs', and 'score') may be null if the favorited item hasn't been submitted to gallery
		
		
		#### Parameters
		
		| Key  | Required | Description                                                                                     |
		|------|----------|-------------------------------------------------------------------------------------------------|
		| page | optional | integer - allows you to set the page number so you don't have to retrieve all the data at once. |
		| sort | optional | 'oldest', or 'newest'. Defaults to 'newest'.                                                    |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/account/{username}/favorites/{page}/{favoritessort}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album_imagevotes15(authorization: str, galleryhash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the vote information about an image
		
		#### Response Model: [Vote](https://api.imgur.com/models/vote)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/{galleryhash}/votes"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album0(authorization: str, albumhash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional information about an album.
		
		#### Response Model: [Album](https://api.imgur.com/models/album)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/album/{albumhash}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def galleryalbum12(authorization: str, galleryhash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional information about an album in the gallery.
		
		#### Response Model: [Gallery Album](https://api.imgur.com/models/gallery_album)"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/album/{galleryhash}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album_imagecomment18(authorization: str, commentid: str, galleryhash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Information about a specific comment. This action also allows any of the additional actions provided in the [Comment Endpoint](https://api.imgur.com/endpoints/comment)."
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/{galleryhash}/comment/{commentid}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def galleryitemtags6(authorization: str, galleryhash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "| Key | Required | Value                  |
		|-----|----------|------------------------|
		| id  | required | ID of the gallery item |"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/{galleryhash}/tags"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gallerytaginfo5(authorization: str, tagname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets metadata about a tag"
    
    """
    url = f"https://imagegur.p.rapidapi.com/3/gallery/tag_info/{tagname}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagegur.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

