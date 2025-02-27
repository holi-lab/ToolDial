import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_covid(q: str, media: str='True', lang: str='en', is_from: str=None, page: int=None, search_in: str=None, to: str=None, topic: str=None, not_sources: str=None, to_rank: int=None, from_rank: int=None, sort_by: str=None, page_size: int=None, sources: str=None, country: str=None, ranked_only: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Highly customizable search through our database of news only on the next specific keywords: 
		`covid`,`covid-19`, `coronavirus`, `коронавирус`, `koronavirüs`, `koronavirüs`, `coronavirüs`, `التاجى`
		All filters and sorting options are allowed."
    q: A word to search on from the list :
`covid`,`covid-19`, `coronavirus`, `коронавирус`, `koronavirüs`, `koronavirüs`, `coronavirüs`, `التاجى`
        media: Adds to the output of the call two more variables: `media` and `media_content`

Media - the main image published with an article 

media_content  - a comma-separated string of all images used in an article
        lang: Specifies the language of the search.  Allowed values are:
 `af`, `ar`, `bg`, `bn`, `ca`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `gu`, `he`,
`hi`, `hr`, `hu`, `id`, `it`, `ja`, `kn`, `ko`, `lt`, `lv`, `mk`, `ml`, `mr`, `ne`, `nl`, `no`, `pa`, `pl`,
`pt`, `ro`, `ru`, `sk`, `sl`, `so`, `sq`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`, `uk`, `ur`, `vi`.
Specifying the language will make your search more relevant
        is_from: From which point in time to start the search. No specific format is required. 
E.g. (`2020/05/01`, `2020-05-01` ,`2020-05-01 12:54:14`, `2020-05-01 4:15am`, `yesterday 2:02 am`, `2020/05/01 12:55 EST`) .
The default timezone is UTC. **Defaults to one week ago at midnight UTC.**
        page: The number of the page. Use it to scroll through the results. Defaults to 1
        search_in: By default, we search what you specified in `q` in both `title` and `summary` of the article. However, you can limit this to either `title` or `summary`
        to: Until which point in time to search for. The default timezone is UTC
        topic: The topic to which you want to restrict the articles of your choice. This parameter is [experimental]. Accepted values are `news`, `sport`, `tech`, `world`, `finance`, `politics`, `business`, `economics`, `entertainment`
`news` usually means a general news feed (main headlines of a website).
> Important: this parameter is [experimental]. Not all news articles are assigned with a `topic`, therefore, we cannot guarantee that 100% of topics talking about technology will be assigned a `tech` label.
> One topic at a time, for example `topic=sport`
        not_sources: One or more sources to be excluded from the search. 
> Comma separated string, for example `not_sources=nytimes.com,cnn.com,wsj.com`
        to_rank: Upper boundary of the rank of news website to filter by
        from_rank: Lowest boundary of the rank of news website to filter by. **Important**: lower rank means that a source is more popular
        sort_by: - `relevancy` — the most relevant results first, 
- `date` — the most recently published results first, 
- `rank` — the results from the highest-ranked sources first
        page_size: How many articles to return per page. Defaults to 50, max is 100
        sources: One or more news resources to filter your search. It should be the normal form of the URL, for example, **[nytimes.com](http://nytimes.com/)**, **[theguardian.com](http://theguardian.com/)**
> Comma separated string, for example `sources=nytimes.com,cnn.com,wsj.com`
        country: The country to which you want to narrow your search. This parameter is [experimental]. We advise you to use it in conjunction with the `lang` parameter. Accepts any [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) codes
        ranked_only: Limit the search only for the sources which are in top 1 million online websites. Defaults to `True` (`False` if you want to turn it off). Unranked sources are assigned a rank that equals to `999999`
        
    """
    url = f"https://covid-19-news.p.rapidapi.com/v1/covid"
    querystring = {'q': q, }
    if media:
        querystring['media'] = media
    if lang:
        querystring['lang'] = lang
    if is_from:
        querystring['from'] = is_from
    if page:
        querystring['page'] = page
    if search_in:
        querystring['search_in'] = search_in
    if to:
        querystring['to'] = to
    if topic:
        querystring['topic'] = topic
    if not_sources:
        querystring['not_sources'] = not_sources
    if to_rank:
        querystring['to_rank'] = to_rank
    if from_rank:
        querystring['from_rank'] = from_rank
    if sort_by:
        querystring['sort_by'] = sort_by
    if page_size:
        querystring['page_size'] = page_size
    if sources:
        querystring['sources'] = sources
    if country:
        querystring['country'] = country
    if ranked_only:
        querystring['ranked_only'] = ranked_only
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

