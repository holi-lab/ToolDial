import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_aggregation(q: str, country: str=None, topic: str=None, sources: str=None, agg_by: str='day', to_rank: int=None, media: str='True', lang: str=None, not_sources: str=None, to: str=None, ranked_only: str=None, is_from: str=None, search_in: str=None, from_rank: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches as `/v1/search` but returns the articles count per hour or day instead of articles' data"
    q: String to search for. Has to be [URL-encoded](https://en.wikipedia.org/wiki/Percent-encoding). 
        country: The country to which you want to narrow your search. This parameter is [experimental]. We advise you to use it in conjunction with the `lang` parameter. Accepts any [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) codes
        topic: The topic to which you want to restrict the articles of your choice. This parameter is [experimental]. Accepted values are `news`, `sport`, `tech`, `world`, `finance`, `politics`, `business`, `economics`, `entertainment`,`beauty`,`travel`,`music`,`food`,`science`
- `news` usually means a general news feed (main headlines of a website).
> Important: this parameter is [experimental]. Not all news articles are assigned with a `topic`, therefore, we cannot guarantee that 100% of topics talking about technology will be assigned a `tech` label.
> One topic at a time, for example, `topic=sport`
        sources: One or more news resources to filter your search. It should be the normal form of the URL, for example, **[nytimes.com](http://nytimes.com/)**, **[theguardian.com](http://theguardian.com/)**
> Comma separated string, for example `sources=nytimes.com,cnn.com,wsj.com`
        agg_by: - `day` — default option. Aggregate results by day. No more than 100 days 
- `hour` — Aggregate results by hour. No more than 100 hours
        to_rank: Upper boundary of the rank of news website to filter by
        media: Adds to the output of the call two more variables: `media` and `media_content`

Media - the main image published with an article 

media_content  - a comma-separated string of all images used in an article
        lang: Specifies the language of the search.  Allowed values are:
`af`, `ar`, `bg`, `bn`, `ca`,`cn`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `gu`, `he`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `kn`, `ko`, `lt`, `lv`, `mk`, `ml`, `mr`, `ne`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `so`, `sq`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`,`tw`, `uk`, `ur`, `vi`. 
Specifying the language will make your search more relevant
        not_sources: One or more sources to be excluded from the search. 
> Comma separated string, for example, `not_sources=nytimes.com,cnn.com,wsj.com`
        to: Until which point in time to search for. The default timezone is UTC
        ranked_only: Limit the search only for the sources which are in top 1 million online websites. Defaults to `True` (`False` if you want to turn it off). Unranked sources are assigned a rank that equals to `999999`
        is_from: From which point in time to start the search. No specific format is required. E.g. (`2020/05/01`, `2020-05-01` ,`2020-05-01 12:54:14`, `2020-05-01 4:15am`, `yesterday 2:02 am`, `2020/05/01 12:55 EST`) Default timezone is UTC. **
The default is set to one week ago at midnight UTC.** 
        search_in: By default, we search what you specified in `q` in both `title` and `summary` of the article. However, you can limit this to either `title` or `summary`
        from_rank: Lowest boundary of the rank of news website to filter by. **Important**: lower rank means that a source is more popular
        
    """
    url = f"https://newscatcher.p.rapidapi.com/v1/aggregation"
    querystring = {'q': q, }
    if country:
        querystring['country'] = country
    if topic:
        querystring['topic'] = topic
    if sources:
        querystring['sources'] = sources
    if agg_by:
        querystring['agg_by'] = agg_by
    if to_rank:
        querystring['to_rank'] = to_rank
    if media:
        querystring['media'] = media
    if lang:
        querystring['lang'] = lang
    if not_sources:
        querystring['not_sources'] = not_sources
    if to:
        querystring['to'] = to
    if ranked_only:
        querystring['ranked_only'] = ranked_only
    if is_from:
        querystring['from'] = is_from
    if search_in:
        querystring['search_in'] = search_in
    if from_rank:
        querystring['from_rank'] = from_rank
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newscatcher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_latest_headlines(lang: str='en', topic: str=None, country: str=None, media: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Always returns the latest 100 articles for any `topic`, `country`, `lang`, or any of those combined"
    lang: Specifies the language of the search.  Allowed values are:
`af`, `ar`, `bg`, `bn`, `ca`,`cn`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `gu`, `he`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `kn`, `ko`, `lt`, `lv`, `mk`, `ml`, `mr`, `ne`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `so`, `sq`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`,`tw`, `uk`, `ur`, `vi`. 
 Specifying the language will make your search more relevant
        topic: The topic to which you want to restrict the articles of your choice. This parameter is [experimental]. Accepted values are `news`, `sport`, `tech`, `world`, `finance`, `politics`, `business`, `economics`, `entertainment`,`beauty`,`travel`,`music`,`food`,`science`
- `news` usually means a general news feed (main headlines of a website).
> Important: this parameter is [experimental]. Not all news articles are assigned with a `topic`, therefore, we cannot guarantee that 100% of topics talking about technology will be assigned a `tech` label.
> One topic at a time, for example `topic=sport`
        country: The country to which you want to narrow your search. This parameter is [experimental]. We advise you to use it in conjunction with the `lang` parameter. Accepts any [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) codes
        media: Adds to the output of the call two more variables: `media` and `media_content`

Media - the main image published with an article 

media_content  - a comma-separated string of all images used in an article
        
    """
    url = f"https://newscatcher.p.rapidapi.com/v1/latest_headlines"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if topic:
        querystring['topic'] = topic
    if country:
        querystring['country'] = country
    if media:
        querystring['media'] = media
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newscatcher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_search_free(q: str, media: str='True', page: str=None, ranked_only: str=None, page_size: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Up to 100 articles per 1 API call even with free Basic Plan.**
		Free search. Only the language filter is allowed."
    q: String to search for. Has to be [URL-encoded](https://en.wikipedia.org/wiki/Percent-encoding)
        media: Adds to the output of the call two more variables: `media` and `media_content`

Media - the main image published with an article 

media_content  - a comma-separated string of all images used in an article
        page: The number of the page. Use it to scroll through the results. Defaults to 1
        ranked_only: Limit the search only for the sources which are in top 1 million online websites. Defaults to `True` (`False` if you want to turn it off). Unranked sources are assigned a rank that equals to `999999`
        page_size: How many articles to return per page. Defaults to 50, max is 100
        lang: Specifies the language of the search.  Allowed values are:
`af`, `ar`, `bg`, `bn`, `ca`,`cn`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `gu`, `he`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `kn`, `ko`, `lt`, `lv`, `mk`, `ml`, `mr`, `ne`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `so`, `sq`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`,`tw`, `uk`, `ur`, `vi`. 
Specifying the language will make your search more relevant
        
    """
    url = f"https://newscatcher.p.rapidapi.com/v1/search_free"
    querystring = {'q': q, }
    if media:
        querystring['media'] = media
    if page:
        querystring['page'] = page
    if ranked_only:
        querystring['ranked_only'] = ranked_only
    if page_size:
        querystring['page_size'] = page_size
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newscatcher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_search(q: str, not_sources: str=None, lang: str='en', search_in: str=None, sort_by: str='relevancy', sources: str=None, to: str=None, country: str=None, media: str='True', topic: str=None, from_rank: int=None, to_rank: int=None, page_size: int=None, page: int=1, ranked_only: str=None, is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Highly customizable search through our database. All filters and sorting options are allowed."
    q: String to search for. Has to be [URL-encoded](https://en.wikipedia.org/wiki/Percent-encoding)
        not_sources: One or more sources to be excluded from the search. 
> Comma separated string, for example `not_sources=nytimes.com,cnn.com,wsj.com`
        lang: Specifies the language of the search.  Allowed values are:
`af`, `ar`, `bg`, `bn`, `ca`,`cn`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `gu`, `he`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `kn`, `ko`, `lt`, `lv`, `mk`, `ml`, `mr`, `ne`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `so`, `sq`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`,`tw`, `uk`, `ur`, `vi`. 
Specifying the language will make your search more relevant
        search_in: By default, we search what you specified in `q` in both `title` and `summary` of the article. However, you can limit this to either `title` or `summary`
        sort_by: - `relevancy` — the most relevant results first, 
- `date` — the most recently published results first, 
- `rank` — the results from the highest-ranked sources first
        sources: One or more news resources to filter your search. It should be the normal form of the URL, for example, **[nytimes.com](http://nytimes.com/)**, **[theguardian.com](http://theguardian.com/)**
> Comma separated string, for example `sources=nytimes.com,cnn.com,wsj.com`
        to: Until which point in time to search for. The default timezone is UTC
        country: The country to which you want to narrow your search. This parameter is [experimental]. We advise you to use it in conjunction with the `lang` parameter. Accepts any [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) codes
        media: Adds to the output of the call two more variables: `media` and `media_content`

Media - the main image published with an article 

media_content  - a comma-separated string of all images used in an article
        topic: The topic to which you want to restrict the articles of your choice. This parameter is [experimental]. Accepted values are `news`, `sport`, `tech`, `world`, `finance`, `politics`, `business`, `economics`, `entertainment`,`beauty`,`travel`,`music`,`food`,`science`
`news` usually means a general news feed (main headlines of a website).
> Important: this parameter is [experimental]. Not all news articles are assigned with a `topic`, therefore, we cannot guarantee that 100% of topics talking about technology will be assigned a `tech` label.
> One topic at a time, for example `topic=sport`
        from_rank: Lowest boundary of the rank of news website to filter by. **Important**: lower rank means that a source is more popular
        to_rank: Upper boundary of the rank of news website to filter by
        page_size: How many articles to return per page. Defaults to 50, max is 100. Not included in Basic plan. 
        page: The number of the page. Use it to scroll through the results. Defaults to 1
        ranked_only: Limit the search only for the sources which are in top 1 million online websites. Defaults to `True` (`False` if you want to turn it off). Unranked sources are assigned a rank that equals to `999999`
        is_from: From which point in time to start the search. No specific format is required. 
E.g. (`2020/05/01`, `2020-05-01` ,`2020-05-01 12:54:14`, `2020-05-01 4:15am`, `yesterday 2:02 am`, `2020/05/01 12:55 EST`) .
The default timezone is UTC. **Defaults to one week ago at midnight UTC.**
        
    """
    url = f"https://newscatcher.p.rapidapi.com/v1/search"
    querystring = {'q': q, }
    if not_sources:
        querystring['not_sources'] = not_sources
    if lang:
        querystring['lang'] = lang
    if search_in:
        querystring['search_in'] = search_in
    if sort_by:
        querystring['sort_by'] = sort_by
    if sources:
        querystring['sources'] = sources
    if to:
        querystring['to'] = to
    if country:
        querystring['country'] = country
    if media:
        querystring['media'] = media
    if topic:
        querystring['topic'] = topic
    if from_rank:
        querystring['from_rank'] = from_rank
    if to_rank:
        querystring['to_rank'] = to_rank
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    if ranked_only:
        querystring['ranked_only'] = ranked_only
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newscatcher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_search_enterprise(q: str, link_begins_with: str=None, sort_by: str='relevancy', topic: str=None, to: str=None, to_rank: int=None, from_rank: int=None, lang: str='en', page_size: int=None, ranked_only: str=None, sources: str=None, search_in: str=None, media: str='True', is_from: str=None, country: str=None, not_sources: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Highly customizable search through our database. All filters and sorting options are allowed."
    q: String to search for. Has to be [URL-encoded](https://en.wikipedia.org/wiki/Percent-encoding)
        link_begins_with: Filter on the prefix of the URL. 
For example: `https://www.nytimes.com/2020/12/25/business` will give you all the articles for the exact date for **business** subject. 
Be aware that every news website has its own structure. Use this parameter only if you want to be precise in your search. In all other cases, use parameters that we provide you for deep search like `topic`, `from`, `to` etc.
        sort_by: - `relevancy` — the most relevant results first, 
- `date` — the most recently published results first, 
- `rank` — the results from the highest-ranked sources first
        topic: The topic to which you want to restrict the articles of your choice. This parameter is [experimental]. Accepted values are `news`, `sport`, `tech`, `world`, `finance`, `politics`, `business`, `economics`, `entertainment`,`beauty`,`travel`,`music`,`food`,`science`
`news` usually means a general news feed (main headlines of a website).
> Important: this parameter is [experimental]. Not all news articles are assigned with a `topic`, therefore, we cannot guarantee that 100% of topics talking about technology will be assigned a `tech` label.
> One topic at a time, for example `topic=sport`
        to: Until which point in time to search for. The default timezone is UTC
        to_rank: Upper boundary of the rank of news website to filter by
        from_rank: Lowest boundary of the rank of news website to filter by. **Important**: lower rank means that a source is more popular
        lang: Specifies the language of the search.  Allowed values are:
`af`, `ar`, `bg`, `bn`, `ca`,`cn`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `gu`, `he`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `kn`, `ko`, `lt`, `lv`, `mk`, `ml`, `mr`, `ne`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `so`, `sq`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`,`tw`, `uk`, `ur`, `vi`. 
Specifying the language will make your search more relevant
        page_size: How many articles to return per page. Defaults to 50, max is 100. Not included in Basic plan. 
        ranked_only: Limit the search only for the sources which are in top 1 million online websites. Defaults to `True` (`False` if you want to turn it off). Unranked sources are assigned a rank that equals to `999999`
        sources: One or more news resources to filter your search. It should be the normal form of the URL, for example, **[nytimes.com](http://nytimes.com/)**, **[theguardian.com](http://theguardian.com/)**
> Comma separated string, for example `sources=nytimes.com,cnn.com,wsj.com`
        search_in: By default, we search what you specified in `q` in both `title` and `summary` of the article. However, you can limit this to either `title` or `summary`
        media: Adds to the output of the call two more variables: `media` and `media_content`

Media - the main image published with an article 

media_content  - a comma-separated string of all images used in an article
        is_from: From which point in time to start the search. No specific format is required. 
E.g. (`2020/05/01`, `2020-05-01` ,`2020-05-01 12:54:14`, `2020-05-01 4:15am`, `yesterday 2:02 am`, `2020/05/01 12:55 EST`) .
The default timezone is UTC. **Defaults to one week ago at midnight UTC.**
        country: The country to which you want to narrow your search. This parameter is [experimental]. We advise you to use it in conjunction with the `lang` parameter. Accepts any [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) codes
        not_sources: One or more sources to be excluded from the search. 
> Comma separated string, for example `not_sources=nytimes.com,cnn.com,wsj.com`
        page: The number of the page. Use it to scroll through the results. Defaults to 1
        
    """
    url = f"https://newscatcher.p.rapidapi.com/v1/search_enterprise"
    querystring = {'q': q, }
    if link_begins_with:
        querystring['link_begins_with'] = link_begins_with
    if sort_by:
        querystring['sort_by'] = sort_by
    if topic:
        querystring['topic'] = topic
    if to:
        querystring['to'] = to
    if to_rank:
        querystring['to_rank'] = to_rank
    if from_rank:
        querystring['from_rank'] = from_rank
    if lang:
        querystring['lang'] = lang
    if page_size:
        querystring['page_size'] = page_size
    if ranked_only:
        querystring['ranked_only'] = ranked_only
    if sources:
        querystring['sources'] = sources
    if search_in:
        querystring['search_in'] = search_in
    if media:
        querystring['media'] = media
    if is_from:
        querystring['from'] = is_from
    if country:
        querystring['country'] = country
    if not_sources:
        querystring['not_sources'] = not_sources
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newscatcher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_sources(lang: str='en', topic: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of supported news websites for any topic, country, language, or any of those combined"
    lang: Specifies the language of the search.  Allowed values are:
`af`, `ar`, `bg`, `bn`, `ca`,`cn`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `gu`, `he`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `kn`, `ko`, `lt`, `lv`, `mk`, `ml`, `mr`, `ne`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `so`, `sq`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`,`tw`, `uk`, `ur`, `vi`. 
 Specifying the language will make your search more relevant
        topic: The topic to which you want to restrict the articles of your choice. This parameter is [experimental]. Accepted values are `news`, `sport`, `tech`, `world`, `finance`, `politics`, `business`, `economics`, `entertainment`,`beauty`,`travel`,`music`,`food`,`science`
- `news` usually means a general news feed (main headlines of a website).
> Important: this parameter is [experimental]. Not all news articles are assigned with a `topic`, therefore, we cannot guarantee that 100% of topics talking about technology will be assigned a `tech` label.
> One topic at a time, for example, `topic=sport`
        country: The country to which you want to narrow your search. This parameter is [experimental]. We advise you to use it in conjunction with the `lang` parameter. Accepts any [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) codes
        
    """
    url = f"https://newscatcher.p.rapidapi.com/v1/sources"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if topic:
        querystring['topic'] = topic
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newscatcher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

