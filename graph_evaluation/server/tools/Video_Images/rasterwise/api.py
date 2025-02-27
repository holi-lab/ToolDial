import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_screenshot(url: str, height: int=None, customjs: str=None, hidecookie: bool=None, hidemsg: bool=None, width: int=None, fullpage: bool=None, webhook: str=None, highlight: str=None, customcss: str=None, preset: str=None, email: str=None, format: str='png', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Handles all the requests to generate screenshots on demand."
    url: URL of the website / page you want to screenshot. Should start with http:// or https:// 
        height: Height in pixels of the viewport when taking the page screenshot. 
        customjs: A custom JS evaluation you want to inject before the capture. If passed we will inject this statement as a header <script> with a just in time model, after all the required operations and just before the capture operation. This is important to have in mind when passing JS that changes the rendering.
        hidecookie: If set to true, we will hide cookie disclaimers that will usually appear as floating boxes or fixed containers. The hiding is not guaranteed but it has a pretty broad coverage and the underlying hiding heuristic is updated weekly.
        hidemsg: If set to true, we will hide message, chat and customer support clients. Currently hides the following clients: Intercom, Drift, Facebook and Tawk (partiallly).
        width: Width in pixels of the viewport when taking the page screenshot.
        fullpage: If set totrue, we will calculate the full height of the website and used it as the height in pixels of the viewport when taking the page screenshot.   Any passed height value will be ignored. 
        webhook: A valid endpoint url that can receive and respond to a POSTrequest (preferably an endpoint that you control).  If set, we will send a POST request with the final response of the original call, to the provided endpoint (webhook listener).  For your convenience we send the response in the body and queryStringParameters of the request.
        highlight: A custom word or phrase you want to highlight. If passed, GetScreenshot will look for that string on the website and highlight all its instances with bright-yellow box.
        customcss: A custom CSS style you want to inject before the capture. If passed we will inject the style declaration as a header <style> just before the capture operation.
        preset: If set, we will control the dimension and user-agent to simulate the preset device or graphics display resolution.  If a preset value is passed, we will ignore other passed dimension parameters.  This parameter can accept any of the following presets:  iphone 5 (iPhone 5)  iphone678 (iPhone 6/7/8)  iphone678_plus (iPhone 6/7/8 +)  iphonex (iPhone X / XS)  pixel2 (Google Pixel 2)  pixel2_xl (Google Pixel 2 XL)  ipad (iPad in Vertical)  ipadpro (iPad Pro Vertical)  hvga (320 x 480)  wvga (480 x 800)  dvga (640 x 960)  wxga_v (768 x 1280)  xga (1024 x 768)  wxga_s (1280 x 800)  wxga_l (1366 x 768)  sxga (1280 x 1024)  wsxga_plus (1680  x 1050)
        email: A valid email address. If set, we will send a formatted email to this email address including the capture image and the details of the capture (capture time and URL).
        format: The file type/format in which you want to get your capture.  It can be either png  or  jpeg.
        
    """
    url = f"https://whoisjuan-rasterwise-v1.p.rapidapi.com/get-screenshot"
    querystring = {'url': url, }
    if height:
        querystring['height'] = height
    if customjs:
        querystring['customjs'] = customjs
    if hidecookie:
        querystring['hidecookie'] = hidecookie
    if hidemsg:
        querystring['hidemsg'] = hidemsg
    if width:
        querystring['width'] = width
    if fullpage:
        querystring['fullpage'] = fullpage
    if webhook:
        querystring['webhook'] = webhook
    if highlight:
        querystring['highlight'] = highlight
    if customcss:
        querystring['customcss'] = customcss
    if preset:
        querystring['preset'] = preset
    if email:
        querystring['email'] = email
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whoisjuan-rasterwise-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

