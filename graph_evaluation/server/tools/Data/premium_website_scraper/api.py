import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def premium_website_scraper_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An overview of the Premium Website Scraper. Please Try Out to get some more information in the response!"
    
    """
    url = f"https://premium-website-scraper.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-website-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scrape_title_title_get(js: bool=None, url: str='https://www.google.de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the website's title."
    
    """
    url = f"https://premium-website-scraper.p.rapidapi.com/title"
    querystring = {}
    if js:
        querystring['js'] = js
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-website-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scrape_url_links_links_get(js: bool=None, url: str='https://www.google.de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extracts all links and associated texts from the requested url to deeper inspect websites and get an overview of links. 
		
		Are you wondering about the API links-results? Often, there is javascript enabled, which limits most content extractors. 
		Use the optional "js" parametter to extract content from websites with js, such as amazon!    "
    
    """
    url = f"https://premium-website-scraper.p.rapidapi.com/links"
    querystring = {}
    if js:
        querystring['js'] = js
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-website-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scrape_headings_headings_get(js: bool=None, url: str='https://www.amazon.de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extracts all headings from the website. 
		
		Are you wondering about the API links-results? Often, there is javascript enabled, which limits most content extractors. 
		Use the optional "js" parametter to extract content from websites with js, such as amazon!    "
    
    """
    url = f"https://premium-website-scraper.p.rapidapi.com/headings"
    querystring = {}
    if js:
        querystring['js'] = js
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-website-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scrape_images_images_get(url: str='https://www.amazon.de', js: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extracts all images from the website and returns the associated links, texts and dimensions. 
		
		Are you wondering about the API links-results? Often, there is javascript enabled, which limits most content extractors. 
		Use the optional "js" parametter to extract content from websites with js, such as amazon!    "
    
    """
    url = f"https://premium-website-scraper.p.rapidapi.com/images"
    querystring = {}
    if url:
        querystring['url'] = url
    if js:
        querystring['js'] = js
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "premium-website-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

