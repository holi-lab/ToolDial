import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_instagram_and_youtube_influencer(current_page: int, posts_minimum: int=None, bio_contains: str=None, posts_maximum: int=None, connector: str=None, country: str=None, category: str=None, city: str=None, engagement_rate_minumum: int=None, followers_maximum: int=None, engagement_rate_maximum: int=None, followers_minimum: int=None, handle_contains: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Instagram & YouTube Influencers using various filters such as Follower Count, Category, Engagement Rate, Post Count, Social Platform, City & Country, Keywords in Bio and influencer handle."
    current_page: navigate through different pages of the desired query data
        posts_minimum: search creator basis minimum posts they have shared
        bio_contains: search creator basis keywords mentioned in their IG bio or YT description
        posts_maximum: search creator basis maximum posts they have shared 
        connector: search creator either from Instagram or YouTube
        country: search creator basis their country
        category: search creator basis their category
        city: search creator basis their city
        engagement_rate_minumum: search creator basis their minimum engagement rate
        followers_maximum: filter creators basis maximum follower/subscriber count
        engagement_rate_maximum: search creator basis their maximum engagement rate
        followers_minimum: filter creators basis minimum follower/subscriber count
        handle_contains: filter creators basis specific text in their handle
        
    """
    url = f"https://ylytic-influencers-api.p.rapidapi.com/ylytic/admin/api/v1/discovery"
    querystring = {'current_page': current_page, }
    if posts_minimum:
        querystring['posts_minimum'] = posts_minimum
    if bio_contains:
        querystring['bio_contains'] = bio_contains
    if posts_maximum:
        querystring['posts_maximum'] = posts_maximum
    if connector:
        querystring['connector'] = connector
    if country:
        querystring['country'] = country
    if category:
        querystring['category'] = category
    if city:
        querystring['city'] = city
    if engagement_rate_minumum:
        querystring['engagement_rate_minumum'] = engagement_rate_minumum
    if followers_maximum:
        querystring['followers_maximum'] = followers_maximum
    if engagement_rate_maximum:
        querystring['engagement_rate_maximum'] = engagement_rate_maximum
    if followers_minimum:
        querystring['followers_minimum'] = followers_minimum
    if handle_contains:
        querystring['handle_contains'] = handle_contains
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ylytic-influencers-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

