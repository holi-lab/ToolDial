import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def capture(html: str=None, json: bool=None, mode: str=None, format: str=None, width: int=None, height: int=None, thumbnail_width: int=None, thumbnail_height: int=None, css: str=None, js: str=None, prescroll: bool=None, delay: int=None, ttl: int=None, fresh: bool=None, user_agent: str=None, accept_language: str=None, element_selector: str=None, headers: str=None, retina: bool=None, base64: bool=None, access_token: str=None, url: str='http://google.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Capture screenshot from any URL or Raw HTML. POST method is also available. Please take a look https://restpack.io/screenshot/docs"
    html: Raw HTML string of a page that you want to capture. Example: <p>Test</p>
        json: Return a JSON response with the resulting image's URL instead of the image itself. Default: false
        mode: Capturing mode. Please see below for details. Default: fullpage Pattern: fullpage | viewport | element
        format: Preferred image output format. If you need a raw html string you can pass html as format Default: png Pattern: jpg | png | pdf | html
        width: Preferred viewport width in pixels. Default: 1280 Min: 320 Max: 2000
        height: Preferred viewport height in pixels. Default: 1024 Min: 160
        thumbnail_width: In case you want a thumbnail image, provide a preferred width. Min: 10 Max: 3000
        thumbnail_height: Preferred thumbnail height, requires thumbnail_width to be set, unbounded if omitted. Min: 10 Max: 3000
        css: Additional CSS string to be injected into the page before render.
        js: Additional JS string to be injected into the page before render.
        prescroll: Force scrolling the webpage before capture. Might help with dynamic loading assets.
        delay: Time in milliseconds to delay capture after page load. Default: 2000 Max: 10000
        ttl: Time in milliseconds for the resulting image to be cached for further requests. Default: 1 day Max: 1 week
        fresh: Force rendering a new screenshot disregarding the cache status. Default: false
        user_agent: Custom user-agent header string for the web request. Default: Chrome Compatible User Agent
        accept_language: Custom accept-language header string for the web request.
        element_selector: A CSS selector to be used with element rendering mode.
        headers: Additional headers seperated with newline Example: X-Test: header\nAccept-Type: html
        retina: Generate retina sized screen capture (2x device pixel ratio). Default: false
        base64: Serialize response file to base64
        access_token: You can provide your token via querystring instead of header.
        url: The URL of web page, including the protocol that you want to capture.
        
    """
    url = f"https://restpack-restpack-screenshot-v2.p.rapidapi.com/capture"
    querystring = {}
    if html:
        querystring['html'] = html
    if json:
        querystring['json'] = json
    if mode:
        querystring['mode'] = mode
    if format:
        querystring['format'] = format
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if thumbnail_width:
        querystring['thumbnail_width'] = thumbnail_width
    if thumbnail_height:
        querystring['thumbnail_height'] = thumbnail_height
    if css:
        querystring['css'] = css
    if js:
        querystring['js'] = js
    if prescroll:
        querystring['prescroll'] = prescroll
    if delay:
        querystring['delay'] = delay
    if ttl:
        querystring['ttl'] = ttl
    if fresh:
        querystring['fresh'] = fresh
    if user_agent:
        querystring['user_agent'] = user_agent
    if accept_language:
        querystring['accept_language'] = accept_language
    if element_selector:
        querystring['element_selector'] = element_selector
    if headers:
        querystring['headers'] = headers
    if retina:
        querystring['retina'] = retina
    if base64:
        querystring['base64'] = base64
    if access_token:
        querystring['access_token'] = access_token
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "restpack-restpack-screenshot-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

