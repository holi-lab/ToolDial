import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def website_screenshot_v1(url: str, type: str=None, ua: str=None, mobile: str=None, nojs: str=None, imageoutputformat: str=None, quality: str=None, delay: str=None, thumbwidth: str=None, mode: str=None, timeout: str=None, height: str=None, scale: str=None, scroll: str=None, landscape: str=None, width: str=None, errorsoutputformat: str=None, fullpage: str=None, touchscreen: str=None, retina: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a screenshot of any web page with one API call (v1)"
    url: The target website's url.
        type: Image output type. Acceptable values: jpg | png pdf. Default: jpg
        ua: The 'User-Agent' header string.
        mobile: If specified, emulates mobile device.
        nojs: If specified, disables JS.
        imageoutputformat: Response output format. Acceptable values: image | base64. Default: image
        quality: Image quality. (only for jpg type). Acceptable values: 40 < quality < 99. Default: jpg
        delay: Custom delay (ms) before screen capture. Acceptable values: 0 < delay < 10000 ms. Default: 250
        thumbwidth: Image thumb width (px). Acceptable values: 50 < thumbWidth < width param value. Default: 0
        mode: fast - waiting for the document.load event. slow - waiting for network idle event. Default: fast
        timeout: Custom timeout (ms) for page loading. Acceptable values: 1000 < timeout < 30000 ms. Default: 15000
        height: Image height (px). Acceptable values: 100 < width < 3000. Default: 600
        scale: deviceScaleFactor value for the emulator. Acceptable values: 0.5 < scale < 4.0. Default: 1.0
        scroll: If specified, scrolls down and up (useful for fullpage screenshots).
        landscape: If specified, renders page in landscape mode (useful for smartphone emulation).
        width: Image width (px). Acceptable values: 100 < width < 3000. Default: 800
        errorsoutputformat: Errors output format. Acceptable values: JSON | XML. Default: JSON
        fullpage: If specified, makes full-page screenshot.
        touchscreen: If specified, emulates device with a touch screens.
        retina: If specified, emulates retina display.
        
    """
    url = f"https://website-screenshot3.p.rapidapi.com/api/v1"
    querystring = {'url': url, }
    if type:
        querystring['type'] = type
    if ua:
        querystring['ua'] = ua
    if mobile:
        querystring['mobile'] = mobile
    if nojs:
        querystring['noJs'] = nojs
    if imageoutputformat:
        querystring['imageOutputFormat'] = imageoutputformat
    if quality:
        querystring['quality'] = quality
    if delay:
        querystring['delay'] = delay
    if thumbwidth:
        querystring['thumbWidth'] = thumbwidth
    if mode:
        querystring['mode'] = mode
    if timeout:
        querystring['timeout'] = timeout
    if height:
        querystring['height'] = height
    if scale:
        querystring['scale'] = scale
    if scroll:
        querystring['scroll'] = scroll
    if landscape:
        querystring['landscape'] = landscape
    if width:
        querystring['width'] = width
    if errorsoutputformat:
        querystring['errorsOutputFormat'] = errorsoutputformat
    if fullpage:
        querystring['fullPage'] = fullpage
    if touchscreen:
        querystring['touchScreen'] = touchscreen
    if retina:
        querystring['retina'] = retina
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "website-screenshot3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

