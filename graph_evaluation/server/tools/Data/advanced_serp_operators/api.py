import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_related_search_terms(query: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Related searches in Google search"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/get-related-search-terms/"
    querystring = {'query': query, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_within_website(query: str, website: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search within a website"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/search-within-website/"
    querystring = {'query': query, 'website': website, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_words_in_title_text_or_url(query: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for words in the title and text or URL"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/search-for-words-in-title-text-or-url/"
    querystring = {'query': query, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_pages_with_keywords(query: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return results where the text appears in the body of the page."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/find-pages-with-keywords/"
    querystring = {'query': query, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_related_keywords(query: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape Google for all the related keywords that it shows at the bottom of the search results page, and show them to you along with the Volume and other SEO metrics in the Related Keyword Tool."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/get-related-keywords/"
    querystring = {'query': query, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_related_questions(query: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The People Also Ask For Keywords Tool. We scrape Google for all the keywords shown in the "people also ask for" boxes, that it shows when you click a link and then go back to the Google SERP."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/get-related-questions/"
    querystring = {'query': query, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_blogs(query: str, max_results: int=10, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can easily find blogs with specific search terms or phrases."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/find-blogs/"
    querystring = {'query': query, }
    if max_results:
        querystring['max_results'] = max_results
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_words_in_url(query: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for words in URL"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/search-for-words-in-url/"
    querystring = {'query': query, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_words_in_title(query: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a web page with certain words in the title"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/search-words-in-title/"
    querystring = {'query': query, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_sites_linking_to_competitions(query: str, website: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find sites linking to competitors"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/find-sites-linking-to-competitions/"
    querystring = {'query': query, 'website': website, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_competition_mentions(keywords: str, domain: str, max_results: int=10, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find PR opportunities by finding competitor mentions"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/find-competition-mentions/"
    querystring = {'keywords': keywords, 'domain': domain, }
    if max_results:
        querystring['max_results'] = max_results
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_pages_linked_to_specific_anchor_text(query: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "finds pages that are linked to any specific anchor text."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/find-pages-linked-to-specific-anchor-text/"
    querystring = {'query': query, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_related_sites(website: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Helps you find sites related to a specified URL."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/search-related-sites/"
    querystring = {'website': website, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_filetype(query: str, filetype: str, max_results: int=10, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search operator helps you to find results for a specific file type."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/search-by-filetype/"
    querystring = {'query': query, 'filetype': filetype, }
    if max_results:
        querystring['max_results'] = max_results
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_links_in_website(website: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for all links in a website"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/search-links-in-website/"
    querystring = {'website': website, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_keywords_in_url(query: str, max_results: int=10, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return only results where the search words are included in the URL"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/find-keywords-in-url/"
    querystring = {'query': query, }
    if max_results:
        querystring['max_results'] = max_results
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_index_block_pages(website: str, blocked_page: str, location: str='us', max_results: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if Indexed Pages You Thought Were Blocked, are indeed not blocked."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/check-index-block-pages/"
    querystring = {'website': website, 'blocked_page': blocked_page, }
    if location:
        querystring['location'] = location
    if max_results:
        querystring['max_results'] = max_results
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_a_range(range_1: str, query: str, range_2: str, max_results: int=10, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search "Android phone" but only show results where the price ranges from $300-$500; also works for dates and other numbers"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/search-range/"
    querystring = {'range_1': range_1, 'query': query, 'range_2': range_2, }
    if max_results:
        querystring['max_results'] = max_results
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def number_indexed_pages(domain: str, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "how many pages (more specifically URLs) a search engine has in its index."
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/number-indexed-pages/"
    querystring = {'domain': domain, }
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_non_secure_pages(domain: str, max_results: int=10, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find non-secure pages (non-https) of a given domain name"
    location: Specify the proxy location for the search. Supported countries. 'US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'
        
    """
    url = f"https://advanced-serp-operators.p.rapidapi.com/find-non-secure-pages/"
    querystring = {'domain': domain, }
    if max_results:
        querystring['max_results'] = max_results
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "advanced-serp-operators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

