import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def main_endpoint_capture_screenshot(url: str, fullpage: int=0, width: str=None, format: str='PNG', secret_key: str=None, css_url: str=None, ttl: int=2592000, placeholder: str=None, user_agent: str=None, accept_lang: str=None, export: str=None, viewport: str='1440x900', delay: str='0', force: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    url: The website URL you want to take a screenshot of
        fullpage: set to "1" in order to capture the full height of the page
        width: define thumbnail width in pixels - e.g. 200
        format: Output formats: PNG, JPEG and GIF
        secret_key: Specify Secret Key - read more at https://screenshotlayer.com/documentation#url_encryption
        css_url: Append a CSS stylesheet URL in order to inject it into the target website
        ttl: Specify caching time (time-to-live) for your screenshot - default: 2592000 seconds (30 days)
        placeholder: Set to 1 in order to use default placeholder image, or set to any image URL in order to use a custom placeholder
        user_agent: Specify HTTP user-agent header
        accept_lang: Specify HTTP accept-language header
        export: Append either S3 Bucket path (format: API_KEY:API_SECRET@bucket) or FTP path (format: ftp://user:password@server)
        viewport: Viewport size - format: "width x height", e.g. 375x667 for iPhone 6
        delay: Specify delay time in seconds - e.g. 3
        force: set to 1 in order to force the API to capture fresh screenshot
        
    """
    url = f"https://apilayer-screenshotlayer-v1.p.rapidapi.com/capture"
    querystring = {'url': url, }
    if fullpage:
        querystring['fullpage'] = fullpage
    if width:
        querystring['width'] = width
    if format:
        querystring['format'] = format
    if secret_key:
        querystring['secret_key'] = secret_key
    if css_url:
        querystring['css_url'] = css_url
    if ttl:
        querystring['ttl'] = ttl
    if placeholder:
        querystring['placeholder'] = placeholder
    if user_agent:
        querystring['user_agent'] = user_agent
    if accept_lang:
        querystring['accept_lang'] = accept_lang
    if export:
        querystring['export'] = export
    if viewport:
        querystring['viewport'] = viewport
    if delay:
        querystring['delay'] = delay
    if force:
        querystring['force'] = force
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-screenshotlayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

