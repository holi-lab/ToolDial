import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def crawler_feeds(output: str, package: str, fmt: str=None, is_id: int=None, limit: int=None, mt: int=None, subject: int=None, politics: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    output: No option defaults to XML.
        package: Crawler feed packages. See readme for a complete list of packages.
        fmt: Field Format. 0 = Default. Provides article title, link, description,  category, publication date, site url/name and article ID. 1 = Above items with meta keywords, meta description, referrer link and geo region id(s). 2 = Above items with article words extract. 3 = All of the above and link to cached copy. Note: fields returned are dependent on availability in article source.
        is_id: To optimize performance and reduce bandwidth, retrieve records greater than this record id.
        limit: Limit the amount of records returned.
        mt: Media Type: 1 = Internet, 2 = Magazine, 3 = News Agencies, 4 = Newspapers, 5 = Radio, 6 = Television
        subject: 13 = Business 4  = Commentary 23 = Crypto Currencies 7  = Entertainment 18 = Environment 16 = Health 17 = Internet: Domain Names 14 = IT 11 = Lead Headline 21 = Libraries/Archives 2  = Local News 10 = National News 3  = Politics 15 = Press Releases 20 = Radio 9  = Science 6  = Sports 22 = Tech:Network Security 5  = Technology 8  = Travel 1  = World News
        politics: Currently assigned only for select US political sites.   9  = Communist, 1  = Conservative, 2  = Democrat, 13 = Leans Left, 12 = Leans Right, 4  = Liberal, 3  = Libertarianism, 5  = Neo-Conservative, 11 = Non-Partisan, 10 = Paleo-Conservative, 6  = Progressive, 7  = Republican, 8  = Socialist
        
    """
    url = f"https://kynast-newslookupcom-crawler-feeds.p.rapidapi.com/feed/{package}"
    querystring = {'output': output, }
    if fmt:
        querystring['fmt'] = fmt
    if is_id:
        querystring['id'] = is_id
    if limit:
        querystring['limit'] = limit
    if mt:
        querystring['mt'] = mt
    if subject:
        querystring['subject'] = subject
    if politics:
        querystring['politics'] = politics
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kynast-newslookupcom-crawler-feeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_feeds(q: str, s: str, package: str, fmt: str=None, limit: str=None, mt: str=None, subject: str=None, politics: str=None, region: str=None, dt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    q: Your query string
        s: Sort. Defaults to relevance, 1 = by date.
        package: Search feed packages. See readme for a complete list of packages.
        fmt: Field Format. 0 = Default. Provides article title, link, description,  category, publication date, site url/name and article ID. 1 = Above items with meta keywords, meta description, referrer link and geo region id(s). 2 = Above items with article words extract. 3 = All of the above and link to cached copy. Note: fields returned are dependent on availability in article source.
        limit: Limit the amount of records returned.
        mt: Media Type: 1 = Internet, 2 = Magazine, 3 = News Agencies, 4 = Newspapers, 5 = Radio, 6 = Television
        subject: 13 = Business 4  = Commentary 23 = Crypto Currencies 7  = Entertainment 18 = Environment 16 = Health 17 = Internet: Domain Names 14 = IT 11 = Lead Headline 21 = Libraries/Archives 2  = Local News 10 = National News 3  = Politics 15 = Press Releases 20 = Radio 9  = Science 6  = Sports 22 = Tech:Network Security 5  = Technology 8  = Travel 1  = World News
        politics: Currently assigned only for select US political sites.   9  = Communist, 1  = Conservative, 2  = Democrat, 13 = Leans Left, 12 = Leans Right, 4  = Liberal, 3  = Libertarianism, 5  = Neo-Conservative, 11 = Non-Partisan, 10 = Paleo-Conservative, 6  = Progressive, 7  = Republican, 8  = Socialist
        region: See documentation for a list of region group ID's.
        dt: Date range. Default is last 60 days.  Last number of days specified in minutes.
        
    """
    url = f"https://kynast-newslookupcom-crawler-feeds.p.rapidapi.com/search/{package}"
    querystring = {'q': q, 's': s, }
    if fmt:
        querystring['fmt'] = fmt
    if limit:
        querystring['limit'] = limit
    if mt:
        querystring['mt'] = mt
    if subject:
        querystring['subject'] = subject
    if politics:
        querystring['politics'] = politics
    if region:
        querystring['region'] = region
    if dt:
        querystring['dt'] = dt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kynast-newslookupcom-crawler-feeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

