import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def snap(url: str, json: bool=None, accuracy: int=None, no_ads: bool=None, no_cookie_banners: bool=None, disable_js: bool=None, image_format: str=None, js_inject: str=None, latitude: str=None, no_tracking: bool=None, longitude: str=None, cookies: str='[]', accept_language: str=None, css_inject: str=None, user_agent: str=None, headers: str='[]', quality: int=None, scale_factor: str=None, wait_for: str=None, page_height: str=None, multi_step_actions: str=None, selectors: str=None, placeholder_url: str=None, page_width: int=None, wait: int=None, hover_on_selected: bool=None, fullpage: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a screenshot immediately. By default it returns the requested image. If the json argument is set to true, it will return a json object containing the direct link and the thumbnail link to the image."
    url: <p>The URL you would like to capture. For example:</p>
      <p>https://pagepixels.com</p>
        json: The snap endpoint returns the image directly by default. When json is set to true, it will return the image's metadata including both the direct and thumbnail links
        accuracy: <p>Provide location information to the browser. You can learn more about the latitude, longitude, and accuracy configuration settings here:</p>
      <p><a href='https://developer.mozilla.org/en-US/docs/Web/API/GeolocationCoordinates' target='_blank'>Learn More</a></p>
        js_inject: Inject any Javascript you need on the page.
        latitude: <p>Provide location information to the browser. You can learn more about the latitude, longitude, and accuracy configuration settings here:</p>
      <p><a href='https://developer.mozilla.org/en-US/docs/Web/API/GeolocationCoordinates' target='_blank'>Learn More</a></p>
        longitude: <p>Provide location information to the browser. You can learn more about the latitude, longitude, and accuracy configuration settings here:</p>
      <p><a href='https://developer.mozilla.org/en-US/docs/Web/API/GeolocationCoordinates' target='_blank'>Learn More</a></p>
        accept_language: <p>Instruct the browser to ask the website to display a specific language. This is great for websites that support multiple languages. You can learn more about this setting here:</p>
      <p><a href='https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Language' target='_blank'>Learn More</a>
        css_inject: Inject any CSS declarations you need into the page.
        user_agent: <p>Give the browser additional information about the type of device that is connecting to the website. You can learn more about the User Agent setting here:</p>
      <p><a href='https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent' target='_blank'>Learn More</a></p>
        quality: <p>Quality defines the JPEG compression, where 1 produces the lowest quality image and 100 creates the highest quality image.  The higher the quality, the larger the image will be.  This setting is ignored if you are producing a PNG image.</p>
        scale_factor: Scale factor 2 produces a Retina display scaled image.
        wait_for: <p>Provide a CSS selector that should appear on the page before the screenshot is captured. For example:</p>
      <p>#id-to-wait-for</p>
        page_height: <p>This is the height of the browser when the screenshot is taken.</p>
        selectors: <p>By default, a screenshot of the entire window is taken. But, you can provide a CSS selector to screenshot only a specific part of the page.  You can learn more about CSS selectors here:</p>
      <p><a href='https://developer.mozilla.org/en-US/docs/Learn/CSS/Building_blocks/Selectors' target='_blank'>Learn More</a></p>
        placeholder_url: <p>The image at the provided URL will be displayed until the first screenshot is captured. The provided URL should return an image.</p>
        page_width: <p>This is the width of the browser when the screenshot is taken.</p>
        wait: <p>How many milliseconds after the webpage has loaded should we wait to capture the screenshot? Maximum wait time is 20 seconds.</p>
        hover_on_selected: <p>If you provided a CSS selector above, you can check this option to have us hover over the object before taking the screenshot.  This gives you the ability to display the :hover styles for the associated element if needed.
        fullpage: <p>By default, screenshots capture only the image within the viewport.  With Full Page selected, the entire page is scrolled and captured.</p>
        
    """
    url = f"https://pagepixels-screenshots.p.rapidapi.com/snap"
    querystring = {'url': url, }
    if json:
        querystring['json'] = json
    if accuracy:
        querystring['accuracy'] = accuracy
    if no_ads:
        querystring['no_ads'] = no_ads
    if no_cookie_banners:
        querystring['no_cookie_banners'] = no_cookie_banners
    if disable_js:
        querystring['disable_js'] = disable_js
    if image_format:
        querystring['image_format'] = image_format
    if js_inject:
        querystring['js_inject'] = js_inject
    if latitude:
        querystring['latitude'] = latitude
    if no_tracking:
        querystring['no_tracking'] = no_tracking
    if longitude:
        querystring['longitude'] = longitude
    if cookies:
        querystring['cookies'] = cookies
    if accept_language:
        querystring['accept_language'] = accept_language
    if css_inject:
        querystring['css_inject'] = css_inject
    if user_agent:
        querystring['user_agent'] = user_agent
    if headers:
        querystring['headers'] = headers
    if quality:
        querystring['quality'] = quality
    if scale_factor:
        querystring['scale_factor'] = scale_factor
    if wait_for:
        querystring['wait_for'] = wait_for
    if page_height:
        querystring['page_height'] = page_height
    if multi_step_actions:
        querystring['multi_step_actions'] = multi_step_actions
    if selectors:
        querystring['selectors'] = selectors
    if placeholder_url:
        querystring['placeholder_url'] = placeholder_url
    if page_width:
        querystring['page_width'] = page_width
    if wait:
        querystring['wait'] = wait
    if hover_on_selected:
        querystring['hover_on_selected'] = hover_on_selected
    if fullpage:
        querystring['fullpage'] = fullpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepixels-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jobs_job_id(job_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The job status endpoint checks if the screenshot associated the job_id returned by the create and capture endpoints is complete."
    job_id: job_id
        
    """
    url = f"https://pagepixels-screenshots.p.rapidapi.com/jobs/{job_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepixels-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshots(before_date: str=None, page: int=None, after_date: str=None, limit: int=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an array of all screenshots captured for the entire account."
    before_date: Return only records created before this date. Date should be a UNIX Timestamp with or without milliseconds.
        page: The page you would like to retrieve.
        after_date: Return only records created after this date. Date should be a UNIX Timestamp with or without milliseconds.
        limit: Limit the total number of records returned by the request.
        order: Can be either ASC or DESC. Records returned in DESC (descending) order will return the most recent records first to the oldest records. Records returned in ASC (ascending) order will return the oldest records first to the newest records
        
    """
    url = f"https://pagepixels-screenshots.p.rapidapi.com/screenshots"
    querystring = {}
    if before_date:
        querystring['before_date'] = before_date
    if page:
        querystring['page'] = page
    if after_date:
        querystring['after_date'] = after_date
    if limit:
        querystring['limit'] = limit
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepixels-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def change_notifications(limit: int=None, before_date: str=None, order: str=None, after_date: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an array of all <a href='https://pagepixels.com/app/documentation#screenshot-change-notifications'>Change Notifications</a> for all screenshot configurations."
    limit: Limit the total number of records returned by the request.
        before_date: Return only records created before this date. Date should be a UNIX Timestamp with or without milliseconds.
        order: Can be either ASC or DESC. Records returned in DESC (descending) order will return the most recent records first to the oldest records. Records returned in ASC (ascending) order will return the oldest records first to the newest records
        after_date: Return only records created after this date. Date should be a UNIX Timestamp with or without milliseconds.
        page: The page you would like to retrieve.
        
    """
    url = f"https://pagepixels-screenshots.p.rapidapi.com/change_notifications"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if before_date:
        querystring['before_date'] = before_date
    if order:
        querystring['order'] = order
    if after_date:
        querystring['after_date'] = after_date
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepixels-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_configs_id_screenshots(is_id: str, order: str=None, after_date: str=None, page: int=None, limit: int=None, before_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an array of all screenshots previously captured by the screenshot configuration."
    id: id
        order: Can be either ASC or DESC. Records returned in DESC (descending) order will return the most recent records first to the oldest records. Records returned in ASC (ascending) order will return the oldest records first to the newest records
        after_date: Return only records created after this date. Date should be a UNIX Timestamp with or without milliseconds.
        page: The page you would like to retrieve.
        limit: Limit the total number of records returned by the request.
        before_date: Return only records created before this date. Date should be a UNIX Timestamp with or without milliseconds.
        
    """
    url = f"https://pagepixels-screenshots.p.rapidapi.com/screenshot_configs/{is_id}/screenshots"
    querystring = {}
    if order:
        querystring['order'] = order
    if after_date:
        querystring['after_date'] = after_date
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if before_date:
        querystring['before_date'] = before_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepixels-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_configs_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the configuration associated with any screenshot configuration created with the create endpoint."
    id: id
        
    """
    url = f"https://pagepixels-screenshots.p.rapidapi.com/screenshot_configs/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepixels-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_configs(after_date: str=None, before_date: str=None, page: int=None, limit: int=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an array of all Screenshot Configurations in your account."
    after_date: Return only records created after this date. Date should be a UNIX Timestamp with or without milliseconds.
        before_date: Return only records created before this date. Date should be a UNIX Timestamp with or without milliseconds.
        page: The page you would like to retrieve.
        limit: Limit the total number of records returned by the request.
        order: Can be either ASC or DESC. Records returned in DESC (descending) order will return the most recent records first to the oldest records. Records returned in ASC (ascending) order will return the oldest records first to the newest records
        
    """
    url = f"https://pagepixels-screenshots.p.rapidapi.com/screenshot_configs"
    querystring = {}
    if after_date:
        querystring['after_date'] = after_date
    if before_date:
        querystring['before_date'] = before_date
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepixels-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screenshot_configs_id_change_notifications(is_id: str, after_date: str=None, order: str=None, page: int=None, before_date: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an array of all <a href='https://pagepixels.com/app/documentation#screenshot-change-notifications'>Change Notifications</a> for a specific screenshot configuration."
    id: id
        after_date: Return only records created after this date. Date should be a UNIX Timestamp with or without milliseconds.
        order: Can be either ASC or DESC. Records returned in DESC (descending) order will return the most recent records first to the oldest records. Records returned in ASC (ascending) order will return the oldest records first to the newest records
        page: The page you would like to retrieve.
        before_date: Return only records created before this date. Date should be a UNIX Timestamp with or without milliseconds.
        limit: Limit the total number of records returned by the request.
        
    """
    url = f"https://pagepixels-screenshots.p.rapidapi.com/screenshot_configs/{is_id}/change_notifications"
    querystring = {}
    if after_date:
        querystring['after_date'] = after_date
    if order:
        querystring['order'] = order
    if page:
        querystring['page'] = page
    if before_date:
        querystring['before_date'] = before_date
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pagepixels-screenshots.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

