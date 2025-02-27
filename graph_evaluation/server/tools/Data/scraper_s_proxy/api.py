import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tor_get(user_agent: str=None, device: str=None, params: str=None, url: str='http://expyuzz4wqqyqhjn.onion/about/history/', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send request to the [Tor network](//www.torproject.org/). Use [Standard GET](//rapidapi.com/scapers-proxy-scapers-proxy-default/api/scrapers-proxy2) instead for better performance and reliability for normal websites. Only recommended to access websites that are only accessible from the Tor network (e.g. websites with a ".onion" top level domain), since this enpoint is slower than other endpoints."
    user_agent: Pass in `user_agent` if the page you are trying to scrape requires a specific user agent. If the page does not require a specific user agent, but a user agent from a type of device using `device` is recommended
        device: Pass in `device` to specify the type of web page you would like to see without needing to specify a user agent. This is recommended as an alternative to using `user_agent ` since it has a higher success rate
        params:  Pass in `params` as json serialized object to specify url query parameters. This is an alternative to adding a query string to the `url` parameter
        
    """
    url = f"https://scrapers-proxy2.p.rapidapi.com/tor"
    querystring = {}
    if user_agent:
        querystring['user_agent'] = user_agent
    if device:
        querystring['device'] = device
    if params:
        querystring['params'] = params
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapers-proxy2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def javascript_rendered_page_get(url: str, session: str=None, user_agent: str=None, country: str=None, device: str=None, click_selector: str=None, params: str=None, wait_ajax: str=None, wait_time: int=10000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Render html using a real browser. Useful for if content is loaded asynchronously or generated dynamically in the browser. JavaScript rendering is usually required to scrape websites that use React, Angular or Vue. For websites that do not need javascript rendering use [Standard GET](//rapidapi.com/scapers-proxy-scapers-proxy-default/api/scrapers-proxy2) instead for better performance and reliability."
    url:  Pass in `url` to specify the url that you want to fetch. If you require  query parameters you can include a query string in the url or specify a json serialized object in the `params` parameter
        session: Pass in `session` to keep cookies and ip address (if necessary) for future requests. You can obtain a session token from the response header `scrapers_proxy_session` after sending a request to the api. Session tokens will expire after 30 seconds of inactivity
        user_agent: Pass in `user_agent` if the page you are trying to scrape requires a specific user agent. If the page does not require a specific user agent, but a user agent from a type of device using `device` is recommended
        country: Pass in `country` for requests that require geolocation to route requests to proxies in specific country. Note: using `country` parameter can increase latency and decrease success rate for certain domains
        device: Pass in `device` to specify the type of web page you would like to see without needing to specify a user agent. This is recommended as an alternative to using `user_agent ` since it has a higher success rate
        click_selector: Pass in `click_selector` as a css selector to specify an element that the browser should click on before  capturing the html of the page
        params:  Pass in `params` as json serialized object to specify url query parameters. This is an alternative to adding a query string to the `url` parameter
        wait_ajax: Pass in `wait_ajax` to specify if the browser should wait for ajax requests to finish before capturing the html of the page.
        wait_time: Pass in `wait_time` to specify the time in milliseconds to wait before capturing the resulting html of the page.
        
    """
    url = f"https://scrapers-proxy2.p.rapidapi.com/javascript"
    querystring = {'url': url, }
    if session:
        querystring['session'] = session
    if user_agent:
        querystring['user_agent'] = user_agent
    if country:
        querystring['country'] = country
    if device:
        querystring['device'] = device
    if click_selector:
        querystring['click_selector'] = click_selector
    if params:
        querystring['params'] = params
    if wait_ajax:
        querystring['wait_ajax'] = wait_ajax
    if wait_time:
        querystring['wait_time'] = wait_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapers-proxy2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standard_get(url: str, device: str=None, country: str=None, session: str=None, params: str=None, user_agent: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Basic proxy GET request"
    url:  Pass in `url` to specify the url that you want to fetch. If you require  query parameters you can include a query string in the url or specify a json serialized object in the `params` parameter
        device: Pass in `device` to specify the type of web page you would like to see without needing to specify a user agent. This is recommended as an alternative to using `user_agent ` since it has a higher success rate
        country: Pass in `country` for requests that require geolocation to route requests to proxies in specific country. Note: using `country` parameter can increase latency and decrease success rate for certain domains
        session: Pass in `session` to keep cookies and ip address (if necessary) for future requests. You can obtain a session token from the response header `scrapers_proxy_session` after sending a request to the api. Session tokens will expire after 30 seconds of inactivity
        params:  Pass in `params` as json serialized object to specify url query parameters. This is an alternative to adding a query string to the `url` parameter
        user_agent: Pass in `user_agent` if the page you are trying to scrape requires a specific user agent. If the page does not require a specific user agent, but a user agent from a type of device using `device` is recommended
        
    """
    url = f"https://scrapers-proxy2.p.rapidapi.com/standard"
    querystring = {'url': url, }
    if device:
        querystring['device'] = device
    if country:
        querystring['country'] = country
    if session:
        querystring['session'] = session
    if params:
        querystring['params'] = params
    if user_agent:
        querystring['user_agent'] = user_agent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapers-proxy2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parser_get(url: str, auto_detect: bool=None, parser: str=None, country: str=None, user_agent: str=None, device: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Automatically parses html into an easily processable json format"
    url:  Pass in `url` to specify the url that you want to fetch. If you require  query parameters you can include a query string in the url or specify a json serialized object in the `params` parameter
        auto_detect: Pass in `auto_detect` to get our system to automatically detect which parser to use.
        parser: Pass in `parser` to specify how to parse the page. For example, pass in `generic-extractor` to extract basic information from any page. For more options please contact support.
        country: Pass in `country` for requests that require geolocation to route requests to proxies in specific country. Note: using `country` parameter can increase latency and decrease success rate for certain domains
        user_agent: Pass in `user_agent` if the page you are trying to scrape requires a specific user agent. If the page does not require a specific user agent, but a user agent from a type of device using `device` is recommended
        device: Pass in `device` to specify the type of web page you would like to see without needing to specify a user agent. This is recommended as an alternative to using `user_agent ` since it has a higher success rate
        
    """
    url = f"https://scrapers-proxy2.p.rapidapi.com/parser"
    querystring = {'url': url, }
    if auto_detect:
        querystring['auto_detect'] = auto_detect
    if parser:
        querystring['parser'] = parser
    if country:
        querystring['country'] = country
    if user_agent:
        querystring['user_agent'] = user_agent
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapers-proxy2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

