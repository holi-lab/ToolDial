import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def take_screenshot(targeturl: str, islandscape: str=None, proxycountry: str=None, isfullyloaded: str=None, clickcount: int=1, fullpage: str=None, clickselector: str=None, hastouch: str=None, clickdelay: int=500, clickbutton: str=None, devicescalefactor: int=1, ismobile: str=None, pagewidth: int=1024, pageheight: int=1024, removables: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "collect all parameteres, load the webpage and take screenshot at the end.
		This API save on a S3 bucket and return the url."
    targeturl: Website url
        islandscape: Specifies if the viewport is in landscape mode.
        isfullyloaded: consider navigation to be finished when there are no more than 0 network connections for at least 500 ms. 
Than take screenshot
        fullpage: take screenshot of the entire website page, from header to footer
        clickselector: This method fetches an element with selector, scrolls it into view if needed, and then uses Page.mouse to click in the center of the element. If there's no element matching selector, the method throws an error.
        hastouch: Specify if the viewport supports touch events.
        clickbutton: Mouse button to be used, left click or right click etc
        devicescalefactor: Specify device scale factor.
        ismobile: Whether the meta viewport tag is taken into account.
        pagewidth: Set browser page width
        pageheight: Set browser page height
        removables: remove divs/html by selector
        
    """
    url = f"https://screenshot-maker.p.rapidapi.com/browser/screenshot/_take"
    querystring = {'targetUrl': targeturl, }
    if islandscape:
        querystring['isLandScape'] = islandscape
    if proxycountry:
        querystring['proxyCountry'] = proxycountry
    if isfullyloaded:
        querystring['isFullyLoaded'] = isfullyloaded
    if clickcount:
        querystring['clickCount'] = clickcount
    if fullpage:
        querystring['fullpage'] = fullpage
    if clickselector:
        querystring['clickSelector'] = clickselector
    if hastouch:
        querystring['hasTouch'] = hastouch
    if clickdelay:
        querystring['clickDelay'] = clickdelay
    if clickbutton:
        querystring['clickButton'] = clickbutton
    if devicescalefactor:
        querystring['deviceScaleFactor'] = devicescalefactor
    if ismobile:
        querystring['isMobile'] = ismobile
    if pagewidth:
        querystring['pageWidth'] = pagewidth
    if pageheight:
        querystring['pageHeight'] = pageheight
    if removables:
        querystring['removables'] = removables
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "screenshot-maker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

