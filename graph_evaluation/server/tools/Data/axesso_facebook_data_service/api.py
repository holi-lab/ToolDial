import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def page_info(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch information for a page like followers and many more"
    
    """
    url = f"https://axesso-facebook-data-service.p.rapidapi.com/fba/facebook-page-info"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-facebook-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def post_details(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch post details for a single post"
    
    """
    url = f"https://axesso-facebook-data-service.p.rapidapi.com/fba/facebook-post-details"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-facebook-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments(feedbackid: str, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch comments for a given feedbackId (returned in the post endpoint). For pagination the optional query parameter "after" can be used by including "comments.data.feedback.page_info.end_cursor" from the previous response."
    
    """
    url = f"https://axesso-facebook-data-service.p.rapidapi.com/fba/facebook-lookup-comments"
    querystring = {'feedbackId': feedbackid, }
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-facebook-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def replies(feedbackid: str, after: str='AQHRl0f7W-KIrlyf4-BzIsbPy7f8d8Rqo569GZjNdn5Gi-F1IpuGuxEAP2daHDv_o4BjQQU1iXqraP3hNj1lKbpn8w', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch comments for for given feedbackId which is located in "data.feedback.display_comments.edges.node.feedback.id" in the post or comments endpoint. For pagination the optional query parameter "after" can be used by including "comments.data.feedback.page_info.end_cursor" from the revious response."
    
    """
    url = f"https://axesso-facebook-data-service.p.rapidapi.com/fba/facebook-lookup-comments"
    querystring = {'feedbackId': feedbackid, }
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-facebook-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def posts(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query posts by providing url e.g. https://www.facebook.com/longhornsteakhouse/posts/.  In the response the field "nextUrl" have to passed to the next request to fetch the next posts."
    
    """
    url = f"https://axesso-facebook-data-service.p.rapidapi.com/fba/facebook-lookup-posts"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-facebook-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

