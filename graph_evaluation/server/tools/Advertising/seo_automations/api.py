import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def extract_top_keywords_phrases_by_density_all_headers(url: str, numphrases: int=20, phraselength: int=5, n: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Analyze any webpage's keyword density for SEO or research. Get the specified number of keywords or phrases you wish, sorted by density and occurrences in a JSON format. The algorithm has been optimized for pages with large amounts of text, as well as advanced cloud infrastracure ensuring accurate and speedy results for your SEO needs. This API uses Puppeteer to scrape even the most advanced SPA ( Single Page Applications )."
    numphrases: Number of phrases to return ( results are sorted by density ). 
Between 1 and 100.
Default is 100.
        phraselength: Number of words in each phrase to return ( results are sorted by density ). 
Between 1 and 10.
Default is 10.
        n: Number of Keywords to return ( results are sorted by density ). 
Between 1 and 100.
Default is 100.
        
    """
    url = f"https://seo-automations.p.rapidapi.com/v1/seo/getKeywordDensity/"
    querystring = {'url': url, }
    if numphrases:
        querystring['numPhrases'] = numphrases
    if phraselength:
        querystring['phraseLength'] = phraselength
    if n:
        querystring['n'] = n
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-automations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_sitemap_xml_as_json(url: str, meta: bool=None, breadcrumbs: bool=None, categories: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Are you looking for an API that can quickly and easily download and parse sitemap.xml files into JSON format? Look no further! Our API allows you to make a simple GET request, passing in the URL of a sitemap.xml file as a parameter. The API will handle the rest, downloading the file, parsing it into a JSON object, and returning the result in the response. And if you need a little extra information to help organize and navigate the URLs in the sitemap, we've got you covered there too. By adding a query parameter to the GET request, you can also include category and breadcrumb information for each URL in the parsed sitemap. Give it a try and see how much easier working with sitemaps can be!"
    
    """
    url = f"https://seo-automations.p.rapidapi.com/v1/seo/fetchsitemap/"
    querystring = {'url': url, }
    if meta:
        querystring['meta'] = meta
    if breadcrumbs:
        querystring['breadcrumbs'] = breadcrumbs
    if categories:
        querystring['categories'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-automations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

