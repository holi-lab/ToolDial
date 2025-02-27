import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def takescreenshotget(screenshoty: int=None, viewportheight: int=1080, html: str=None, screenshotx: int=None, selectortoscreenshot: str=None, viewportwidth: int=1920, responsetype: str='file', readyevent: str='load', screenshotwidth: int=None, url: str='https://google.com', cookies: str=None, httpheaders: str=None, omitbackground: bool=None, prerendercode: str=None, format: str='image/png', useragent: str=None, screenshotheight: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receives a URL, takes the screenshot and return the image"
    screenshoty: The screenshot Y clip position (leave blank if you want full page screenshot)
        viewportheight: The screenshot website desired viewport height
        html: A html code to use instead of a page to take the screenshot from (You are required to send a "html" OR a "url")
        screenshotx: The screenshot X clip position (leave blank if you want full page screenshot)
        selectortoscreenshot: CSS Selector to screenshot in the page (defaults to the entire page)
        viewportwidth: The screenshot website desired viewport height
        responsetype: Which way should the response be received
        readyevent: Which browser event to wait before taking the screenshot.
        screenshotwidth: The screenshot clip width (leave blank if you want full page screenshot)
        url: The website to take the screenshot
        cookies: A JSON Object with cookies to be used in the requested page. It should have a pattern of {"Header Name": "Header Value"}
        httpheaders: A JSON Object with httpHeaders to be sent to the requested page. It should have a pattern of {"Header Name": "Header Value"}
        omitbackground: Turn non colored parts of the background transparent in the screenshot
        prerendercode: Code to execute before taking the screenshot
        format: Which image format should be returned
        useragent: Custom User Agent to Simulate
        screenshotheight: The screenshot clip height (leave blank if you want full page screenshot)
        
    """
    url = f"https://screenshotty1.p.rapidapi.com/api/v1/screenshot"
    querystring = {}
    if screenshoty:
        querystring['screenshotY'] = screenshoty
    if viewportheight:
        querystring['viewportHeight'] = viewportheight
    if html:
        querystring['html'] = html
    if screenshotx:
        querystring['screenshotX'] = screenshotx
    if selectortoscreenshot:
        querystring['selectorToScreenshot'] = selectortoscreenshot
    if viewportwidth:
        querystring['viewportWidth'] = viewportwidth
    if responsetype:
        querystring['responseType'] = responsetype
    if readyevent:
        querystring['readyEvent'] = readyevent
    if screenshotwidth:
        querystring['screenshotWidth'] = screenshotwidth
    if url:
        querystring['url'] = url
    if cookies:
        querystring['cookies'] = cookies
    if httpheaders:
        querystring['httpHeaders'] = httpheaders
    if omitbackground:
        querystring['omitBackground'] = omitbackground
    if prerendercode:
        querystring['preRenderCode'] = prerendercode
    if format:
        querystring['format'] = format
    if useragent:
        querystring['userAgent'] = useragent
    if screenshotheight:
        querystring['screenshotHeight'] = screenshotheight
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "screenshotty1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

