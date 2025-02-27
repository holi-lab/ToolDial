import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def request(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a CURL request to any URL. (does not render JS but much faster)"
    
    """
    url = f"https://scrape-web-simple-api-for-scraping-any-website.p.rapidapi.com/curl-request"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrape-web-simple-api-for-scraping-any-website.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def browser_request(url: str, wait_browser: str=None, screenshot_full_page: bool=None, wait: int=None, wait_for: str=None, screenshot_selector: str=None, block_images: bool=None, block_css: bool=None, return_page_source: bool=None, window_height: int=1080, window_width: int=1920, pdf: bool=None, screenshot: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a request to any URL with a browser (renders Javascript)"
    wait_browser: This advanced parameter tells the browser to wait until certain network condition are met.

It can take 4 different values:

domcontentloaded (default): Wait until the DOM is loaded
load: Wait until the page is fully loaded
networkidle0: Wait until there are no more than 0 network connections for at least 500 ms
networkidle2: Wait until there are no more than 2 network connections for at least 500 ms

For example, to wait until the page is fully loaded before getting the results, you can use wait_browser=load.
        screenshot_full_page: By default, the screenshot will only capture the portion of the page that is visible in the browser's viewport.

If you need a screenshot of the full page from the target website use screenshot_full_page=true

If you need to change the size of the browser's viewport before taking a screenshot you can do it using window_width and window_height parameters.

Screenshots are kept for 30 days and then deleted.
        wait: Wait a fixed amount of time

Some code-heavy websites need time to fully render. To wait before it returns the fully rendered HTML, use the wait parameter with a value in milliseconds between 0 and 35000.

The headless browsers will then wait the duration of the time set in milliseconds before returning the page's HTML.
        wait_for: It's sometimes necessary to wait for a particular element to appear in the DOM before browser returns the HTML content.

Our headless browsers will wait for the CSS / Xpath selector passed in the parameter before returning the HTML.

Example/
wait_for=.loading-page
        screenshot_selector: By default, the screenshot will only capture the portion of the page that is visible in the browser's viewport.

If you need to screenshot a particular area of the page, you can use screenshot_selector=<CSS_selector> where <CSS_selector> is the CSS selector of the area you want to capture.

Screenshots are kept for 30 days and then deleted.
        block_images: By default, and to speed up requests, all images and CSS are blocked. To enable images,  block_images=false can be used. 

        block_css: By default, and to speed up requests, all images and CSS are blocked. To enable css,  block_css=false can be used. 

        return_page_source: To have HTML returned by the server and unaltered by the browser (before the JavaScript execution), use return_page_source=true

        window_height: If you need to change the dimension of the browser's viewport (window) when scraping the target page you can use the window_width and window_height parameters.
        window_width: If you need to change the dimension of the browser's viewport (window) when scraping the target page you can use the window_width and window_height parameters.
        pdf: Returns a PDF for the rendered webpage. PDF is kept for 30 days and then deleted.
        screenshot: If you want to get a screenshot of the page you want to scrape, use the screenshot=true parameter.

Screenshots are kept for 30 days and then deleted.
        
    """
    url = f"https://scrape-web-simple-api-for-scraping-any-website.p.rapidapi.com/"
    querystring = {'url': url, }
    if wait_browser:
        querystring['wait_browser'] = wait_browser
    if screenshot_full_page:
        querystring['screenshot_full_page'] = screenshot_full_page
    if wait:
        querystring['wait'] = wait
    if wait_for:
        querystring['wait_for'] = wait_for
    if screenshot_selector:
        querystring['screenshot_selector'] = screenshot_selector
    if block_images:
        querystring['block_images'] = block_images
    if block_css:
        querystring['block_css'] = block_css
    if return_page_source:
        querystring['return_page_source'] = return_page_source
    if window_height:
        querystring['window_height'] = window_height
    if window_width:
        querystring['window_width'] = window_width
    if pdf:
        querystring['pdf'] = pdf
    if screenshot:
        querystring['screenshot'] = screenshot
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrape-web-simple-api-for-scraping-any-website.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

