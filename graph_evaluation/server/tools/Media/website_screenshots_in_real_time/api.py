import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def account_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about your account."
    
    """
    url = f"https://browshot.p.rapidapi.com/account/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_delete(is_id: int, data: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete details of your screenshots to remove any confidential information."
    id: Screenshot ID
        data: data to remove: image (default), url, metadata, all
        
    """
    url = f"https://browshot.p.rapidapi.com/screenshot/delete"
    querystring = {'id': is_id, }
    if data:
        querystring['data'] = data
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_create(url: str, instance_id: int=None, size: str='screen', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Screenshots requests to private and shared instances require a positive balance."
    url: URL of the page to get a screenshot for
        instance_id: instance ID to use (default: 12)
        size: screenshot size: "screen" (default) or "page"
        
    """
    url = f"https://browshot.p.rapidapi.com/screenshot/create"
    querystring = {'url': url, }
    if instance_id:
        querystring['instance_id'] = instance_id
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instance_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of instances available."
    
    """
    url = f"https://browshot.p.rapidapi.com/instance/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def browser_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all predefined web browsers."
    
    """
    url = f"https://browshot.p.rapidapi.com/browser/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def browser_info(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of specific browser."
    id: Browser ID
        
    """
    url = f"https://browshot.p.rapidapi.com/browser/info"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_thumbnail(is_id: int, width: int=1024, height: int=None, scale: int=1, zoom: int=50, ratio: str='fill', left: int=0, right: int=768, top: int=0, bottom: int=768, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sends back the thumbnail as a PNG file."
    id: Screenshot ID
        width: Width of the thumbnail
        height: Height of the thumbnail
        scale: Scale of the thumbnail
        zoom: 1-100: zoom 1 to 100 percents
        ratio: <fit|fill> (default: fit). Use fit to keep the original page ration, and fill to get a thumbnail for the exact width and height specified.
        left: (default: 0) left edge of the area to be cropped
        right: (default: 0) right edge of the area to be cropped
        top: (default: 0) top edge of the area to be cropped
        bottom: right edge of the area to be cropped
        
    """
    url = f"https://browshot.p.rapidapi.com/screenshot/thumbnail"
    querystring = {'id': is_id, }
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if scale:
        querystring['scale'] = scale
    if zoom:
        querystring['zoom'] = zoom
    if ratio:
        querystring['ratio'] = ratio
    if left:
        querystring['left'] = left
    if right:
        querystring['right'] = right
    if top:
        querystring['top'] = top
    if bottom:
        querystring['bottom'] = bottom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_list(limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about the last 100 screenshots requested."
    limit: number of screenshots' information to return (maximum: 100)
        
    """
    url = f"https://browshot.p.rapidapi.com/screenshot/list"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_share(is_id: int, note: str='this+is+my+screenshot', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make your screenshots public, add notes, and share it with your friends and colleagues."
    id: Screenshot ID
        note: Mote to add on the sharing page
        
    """
    url = f"https://browshot.p.rapidapi.com/screenshot/share"
    querystring = {'id': is_id, }
    if note:
        querystring['note'] = note
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instance_info(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of a specific instance"
    id: Instance ID
        
    """
    url = f"https://browshot.p.rapidapi.com/instance/info"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple(url: str, instance_id: int=12, width: int=128, height: int=256, size: str='page', cache: int=60, referer: str='http://www.google.com/', post_data: str='login=foo&password=bar', cookie: str='session_id=11111111111', delay: int=5, flash_delay: int=10, script: str='http://mydomain.com/myscript/script.js', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a screenshot or thumbnail in one request. make sure to follow all 302 redirections."
    url: Url to take screenshot of
        instance_id: Browser ID to use
        width: Width of the thumbnail
        height: Height of the thumbnail
        size: Screenshot size: "screen" (default) or "page"
        cache: Use a previous screenshot (same URL, same instance) if it was done within <cache> seconds. The default value is 24hours. Specify cache=0 if you want a new screenshot.
        referer: Use a custom referrer
        post_data: Send a POST requests with post_data, useful for filling out forms
        cookie: Set a cookie for the URL requested
        delay: 0-10 (default: 5): number of seconds to wait after the page has loaded. This is used to let JavaScript run longer before taking the screenshot. Use delay=0 to take screenshots faster.
        flash_delay: 0-10 (default: 10): number of seconds to wait after the page has loaded if Flash elements are present. Use flash_delay=0 to take screenshots faster.
        script: URL of javascript file to execute after the page load event.
        
    """
    url = f"https://browshot.p.rapidapi.com/simple"
    querystring = {'url': url, }
    if instance_id:
        querystring['instance_id'] = instance_id
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if size:
        querystring['size'] = size
    if cache:
        querystring['cache'] = cache
    if referer:
        querystring['referer'] = referer
    if post_data:
        querystring['post_data'] = post_data
    if cookie:
        querystring['cookie'] = cookie
    if delay:
        querystring['delay'] = delay
    if flash_delay:
        querystring['flash_delay'] = flash_delay
    if script:
        querystring['script'] = script
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_info(is_id: int, details: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Once a screenshot has been requested, its status must be checked until it is either "error" or "finished"."
    id: Screenshot ID
        details: 0-3 (default: 2): level of details about the screenshot and the page.
        
    """
    url = f"https://browshot.p.rapidapi.com/screenshot/info"
    querystring = {'id': is_id, }
    if details:
        querystring['details'] = details
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "browshot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

