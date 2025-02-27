import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def restfullink_get(p2i_url: str, p2i_device: str='6', p2i_size: str='300x300', p2i_screen: str='1024x768', p2i_fullpage: str='0', p2i_imageformat: str='jpg', p2i_wait: str='10', p2i_refresh: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    p2i_url: URL of target page. Do not need to be encode. Valid inputs: www.google.com google.com http://www.google.com https://www.google.com
        p2i_device: Our system can generate the screenshot for those smart phones and desktops. Please pass it to server as needed. Following are devices we supported: 0 - iPhone4, 1 - iPhone5, 2 - Android, 3 - WinPhone, 4 - iPad, 5 - Android Pad, 6 - Desktop
        p2i_size: The thumbnail file size you want to get. Not the screen size of device. If you want a big image, you need enlarge the width and height. If you need a small one, you need reduce the width and height. But it will take some more time to download a big image than a small one. Format: widthxheight e.g. 300x300  We predefined the size for those devices for most usage: Mobile Phones: 320x480 Tablets: 512x384 Desktop: 640x400  Advanced information: 1) If you only know the fixed width or height of the images you want to get, you can just pass the width or height to us by following format: 512x0 or 0x512. We will resize the image to fit that width/height and make sure it is the biggest image we can generate. 2) The screenshot image will be resized to fit the width and height without stretching
        p2i_screen: The screen size of device you want to simulate. Format: screen_widthxscreen_height e.g. 1024x768 We predefined the size for those devices for most usage: Mobile: 320x480 Mobile iPhone5: 320x500 Tablets: 1024x768 Desktop: 1280x800 Advanced information: 1) The screen size should not be less than image size.
        p2i_fullpage: Full page or just the screen area. If the value is 1 or true, we will take the full page screenshot. If the value is 0 or false, we will take the screen area only.
        p2i_imageformat: png or jpg We strongly recommend jpg since most jpg file is smaller than png.
        p2i_wait: How many second wait after page load complete. System will not accept the value less than 30 seconds.
        p2i_refresh: We will cache the screenshot result for 24 hours. If you does not want to use the cache, please set this value to 1.
        
    """
    url = f"https://p2i.p.rapidapi.com/restfullink"
    querystring = {'p2i_url': p2i_url, }
    if p2i_device:
        querystring['p2i_device'] = p2i_device
    if p2i_size:
        querystring['p2i_size'] = p2i_size
    if p2i_screen:
        querystring['p2i_screen'] = p2i_screen
    if p2i_fullpage:
        querystring['p2i_fullpage'] = p2i_fullpage
    if p2i_imageformat:
        querystring['p2i_imageformat'] = p2i_imageformat
    if p2i_wait:
        querystring['p2i_wait'] = p2i_wait
    if p2i_refresh:
        querystring['p2i_refresh'] = p2i_refresh
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "p2i.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

