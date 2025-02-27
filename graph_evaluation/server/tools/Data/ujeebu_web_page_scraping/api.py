import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_1_scrape(url: str, response_type: str='html', js: bool=None, block_ads: bool=None, json: bool=None, useragent: str='Mozilla/5.0 (Windows NT 10.0; Trident/7.0; rv:11.0) like Gecko', window_width: int=1366, screenshot_fullpage: bool=None, window_height: int=768, device: str='desktop', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the html source of a rendered web page. Generates PDF or Screenshot of a web page."
    url: URL to scrape
        response_type: The response returned by the endpoint depends on the 'response_type' and 'json' parameters. Response can be either a byte array in the case of 'pdf' and 'screenshot', text when response_type='raw' or 'html' (default), or JSON when json=1. response_type possible values are as follows:
- 'html': returns the html code of the page . If js = 1 it will first execute JavaScript.
- 'raw': returns the source html (or file content if URL is not an HTML page) as is from the URL without running JavaScript. js=1 is ignored.
- 'pdf': converts page to the PDF format and returns the PDF binary data.
If the json parameter is set to 'true' a JSON response is returned with the base64 encoded value of the pdf file.
- 'screenshot': produces a screenshot of the URL in PNG format and returns the binary data. If screenshot_fullpage is set to 'true', takes a screenshot of the full page. If set to 'false', takes a screenshot of the visible viewport only. If the json parameter is set to 'true', a JSON response is returned with the base64 encoded value of the image file.

        js: indicates whether to execute JavaScript or not. Default =0
        block_ads: whether to block ads or not
        json: when set to true, returns a JSON response instead of raw content as specified by 'response_type'. Default = 0.
        useragent: The Scrape API will by default send its own user agent header (Chrome's headless browser header), but if you want to use a different user agent you need to set parameter 'useragent'
        window_width: indicates browser viewport width.
        screenshot_fullpage: when 'content_type' is 'screenshot', whether to take a screenshot of the full page or just the visible viewport
        window_height: browser viewport height
        device: indicates type of device to use to render page. Possible values: 'desktop', 'mobile'. Default='desktop'
        
    """
    url = f"https://ujeebu-web-page-scraping.p.rapidapi.com/v1.1/scrape"
    querystring = {'url': url, }
    if response_type:
        querystring['response_type'] = response_type
    if js:
        querystring['js'] = js
    if block_ads:
        querystring['block_ads'] = block_ads
    if json:
        querystring['json'] = json
    if useragent:
        querystring['useragent'] = useragent
    if window_width:
        querystring['window_width'] = window_width
    if screenshot_fullpage:
        querystring['screenshot_fullpage'] = screenshot_fullpage
    if window_height:
        querystring['window_height'] = window_height
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ujeebu-web-page-scraping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

