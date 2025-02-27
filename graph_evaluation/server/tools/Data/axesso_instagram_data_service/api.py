import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def comments(shortcode: str, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch comments for a given post. Therefore the shortcode needs to be passed which is returned in the "node" array from posts endpoint. If in the first response "data.shortcode_media.edge_media_to_parent_comment.page_info.has_next_page" equals true, then more comments exists and it can be fetched by passing "data.shortcode_media.edge_media_to_parent_comment.page_info.end_cursor" to the after param in the next request."
    
    """
    url = f"https://axesso-instagram-data-service1.p.rapidapi.com/ins/instagram-lookup-comments"
    querystring = {'shortCode': shortcode, }
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-instagram-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accountinfo(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch data for a give account. The response includes the field "id which is required for further requests e.g. posts, comments and replies and needs to be passed to query param userId. This endpoint needs the sessionid to work."
    
    """
    url = f"https://axesso-instagram-data-service1.p.rapidapi.com/ins/instagram-account-info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-instagram-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts(userid: str, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts for a give user. This endpoints needs the userid which can be obtained from the account-info endpoint. For pagination the field data.user.edge_owner_to_timeline_media.page_info.end_cursor from the response needs to be passed to after param in the next request."
    
    """
    url = f"https://axesso-instagram-data-service1.p.rapidapi.com/ins/instagram-lookup-posts"
    querystring = {'userId': userid, }
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-instagram-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replies(after: str, commentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch replies for a given commentId.  CommentId can be obtained from the comments endpoint response in field node.id. Also the field edge_threaded_comments.page_info.end_cursor needs to be passed for the given comment which can also be found in the response from the comments endpoint."
    
    """
    url = f"https://axesso-instagram-data-service1.p.rapidapi.com/ins/instagram-lookup-replies"
    querystring = {'after': after, 'commentId': commentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-instagram-data-service1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

