import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_list_of_latest_profiles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the top 100 crypto projects added to on isthiscoinascam.com. Ordered by date added."
    
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/top/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_most_watched_profiles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the most watched 100 crypto projects on isthiscoinascam.com over the past 7 days. Ordered by most most watched first."
    
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/top/watched"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_highest_community_rated_profiles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the most highly community rated crypto projects on isthiscoinascam.com over the past 7 days. Ordered by highest rating first."
    
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/top/rated"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_platform_by_slug(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific platform by slug"
    slug: Platform Slug
        
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/platforms/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_profile_by_slug(slug: str, explorers: bool=None, community: bool=None, repo: bool=None, contracts: bool=None, news: bool=None, flags: bool=None, exchanges: bool=None, links: bool=None, tags: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific coin profile by slug"
    slug: Slug of Coin
        explorers: 'true' if you want to display the list of explorers
        community: 'true' if you want to display the community metrics related to this coin
        repo: 'true' if you want to display the source code repo stats related to this coin
        contracts: 'true' if you want to display the smart contracts and audit data related to this coin
        news: 'true' if you want to display the latest 5 news stories related to this coin
        flags: 'true' if you want to display the red flags related to this coin
        exchanges: 'true' if you want to display the list of exchanges
        links: 'true' if you want to display the link to social media and project websites and artifacts
        tags: 'true' if you want to display the tags related to this coin
        
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/profiles/{slug}"
    querystring = {}
    if explorers:
        querystring['explorers'] = explorers
    if community:
        querystring['community'] = community
    if repo:
        querystring['repo'] = repo
    if contracts:
        querystring['contracts'] = contracts
    if news:
        querystring['news'] = news
    if flags:
        querystring['flags'] = flags
    if exchanges:
        querystring['exchanges'] = exchanges
    if links:
        querystring['links'] = links
    if tags:
        querystring['tags'] = tags
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_list_of_trending_profiles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of the top 100 crypto projects currently trending on isthiscoinascam.com added to the site. Ordered by most popular first."
    
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/top/trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_profiles(page: int=1, community: bool=None, name: str='bitcoin', repo: bool=None, explorers: bool=None, flags: bool=None, symbol: str='BTC,ETH', exchanges: bool=None, slug: str='bitcoin,ethereum', tags: bool=None, limit: int=10, contracts: bool=None, links: bool=None, news: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of profiles. You can search slug, name and symbol"
    page: the page number which to start from
        community: 'true' if you want to display the community metrics related to this coin
        name: search all profile names. Search for more than 1 name by using a comma seperated list.
        repo: 'true' if you want to display the source code repo stats related to this coin
        explorers: 'true' if you want to display the list of explorers
        flags: 'true' if you want to display the red flags related to this coin
        symbol: search all profile symbols. Search for more than 1 slug by using a comma seperated list.
        exchanges: 'true' if you want to display the list of exchanges
        slug: search all profile slugs. Search for more than 1 slug by using a comma seperated list.
        tags: 'true' if you want to display the tags related to this coin
        limit: limit the number of records returned
        contracts: 'true' if you want to display the smart contracts and audit data related to this coin
        links: 'true' if you want to display the link to social media and project websites and artifacts
        news: 'true' if you want to display the latest 5 news stories related to this coin
        
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/profiles"
    querystring = {}
    if page:
        querystring['page'] = page
    if community:
        querystring['community'] = community
    if name:
        querystring['name'] = name
    if repo:
        querystring['repo'] = repo
    if explorers:
        querystring['explorers'] = explorers
    if flags:
        querystring['flags'] = flags
    if symbol:
        querystring['symbol'] = symbol
    if exchanges:
        querystring['exchanges'] = exchanges
    if slug:
        querystring['slug'] = slug
    if tags:
        querystring['tags'] = tags
    if limit:
        querystring['limit'] = limit
    if contracts:
        querystring['contracts'] = contracts
    if links:
        querystring['links'] = links
    if news:
        querystring['news'] = news
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_by_slug(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific coin by slug"
    slug: Coin Slug
        
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/coins/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_platforms(symbol: str='ETH', limit: int=10, name: str='ethereum', page: int=10, slug: str='ethereum', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of platforms. You can search slug, name and symbol. You can perform wildcard searches"
    symbol: search all platform symbolds. Wildcard searches can be performed by using the * character e.g. symbol=*TH. You can search for more than one item at a time by sending a comma seperated list e.g. symbol=TOK,*TH
        limit: limit the number of records returned
        name: search all platform names. Wildcard searches can be performed by using the * character e.g. name=*platform. You can search for more than one item at a time by sending a comma seperated list e.g. name=ethereum,*token
        page: the page number which to start from
        slug: search all platform slugs. Wildcard searches can be performed by using the * character e.g. slug=*token. You can search for more than one item at a time by sending a comma seperated list e.g. slug=ethereum,*token
        
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/platforms"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if limit:
        querystring['limit'] = limit
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    if slug:
        querystring['slug'] = slug
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_coins(symbol: str=None, name: str=None, page: int=1, slug: str='bitcoin,ethereum', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of coins. You can search slug, name and symbol. You can perform wildcard searches"
    symbol: search all coin symbols. Wildcard searches can be performed by using the * character e.g. symbol=*BTC. You can search for more than one item at a time by sending a comma seperated list e.g. symbol=BTC,*TH
        name: search all coin names. Wildcard searches can be performed by using the * character e.g. name=*coin. You can search for more than one item at a time by sending a comma seperated list e.g. name=bitcoin,*eth
        page: the page number which to start from
        slug: search all coin slugs. Wildcard searches can be performed by using the * character e.g. slug=*coin. You can search for more than one item at a time by sending a comma seperated list e.g. slug=bitcoin,*eth
        limit: limit the number of records returned
        
    """
    url = f"https://is-this-coin-a-scam.p.rapidapi.com/coins"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    if slug:
        querystring['slug'] = slug
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "is-this-coin-a-scam.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

