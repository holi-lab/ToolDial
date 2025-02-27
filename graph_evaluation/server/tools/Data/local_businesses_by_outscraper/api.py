import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_places(query: str, fields: str=None, region: str='US', language: str='en', skipplaces: int=0, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns places from Places Maps based on a given search query (or many queries).
		
		The results from searches are the same as you would see by visiting a regular Maps site. However, in most cases, it's recommended to use locations inside queries (e.g., bars, NY, USA) as the IP addresses of Outscraper's servers might be located in different countries.
		
		This endpoint is optimized for fast responses and can be used as a real-time API."
    query: The parameter defines the query you want to search. You can use anything that you would use on a regular Maps site. Additionally, you can use feature_id, place_id, or CID. The example of valid queries: `Real estate agency, Rome, Italy`, `The NoMad Restaurant, NY, USA`, `restaurants, Brooklyn 11203`, `0x886916e8bc273979:0x5141fcb11460b226,` `ChIJrZhup4lZwokRUr_5sLoFlDw`, etc.

It supports batching by sending arrays with up to 25 queries (e.g., `query=text1&query=text2&query=text3`). It allows multiple queries to be sent in one request and save on network latency time.

You might want to check out [the web application](https://app.outscraper.com/googleMaps) to play with locations and categories that we would suggest.

        fields: The parameter defines which fields you want to include with each item returned in the response. By default, it returns all fields. Use `&fields=query,name` to return only the specific ones.
        region: The parameter specifies the country to use.

Available countries: `AF`, `AL`, `DZ`, `AS`, `AD`, `AO`, `AI`, `AG`, `AR`, `AM`, `AU`, `AT`, `AZ`, `BS`, `BH`, `BD`, `BY`, `BE`, `BZ`, `BJ`, `BT`, `BO`, `BA`, `BW`, `BR`, `VG`, `BN`, `BG`, `BF`, `BI`, `KH`, `CM`, `CA`, `CV`, `CF`, `TD`, `CL`, `CN`, `CO`, `CG`, `CD`, `CK`, `CR`, `CI`, `HR`, `CU`, `CY`, `CZ`, `DK`, `DJ`, `DM`, `DO`, `EC`, `EG`, `SV`, `EE`, `ET`, `FJ`, `FI`, `FR`, `GA`, `GM`, `GE`, `DE`, `GH`, `GI`, `GR`, `GL`, `GT`, `GG`, `GY`, `HT`, `HN`, `HK`, `HU`, `IS`, `IN`, `ID`, `IQ`, `IE`, `IM`, `IL`, `IT`, `JM`, `JP`, `JE`, `JO`, `KZ`, `KE`, `KI`, `KW`, `KG`, `LA`, `LV`, `LB`, `LS`, `LY`, `LI`, `LT`, `LU`, `MG`, `MW`, `MY`, `MV`, `ML`, `MT`, `MU`, `MX`, `FM`, `MD`, `MN`, `ME`, `MS`, `MA`, `MQ`, `MZ`, `MM`, `NA`, `NR`, `NP`, `NL`, `NZ`, `NI`, `NE`, `NG`, `NU`, `MK`, `NO`, `OM`, `PK`, `PS`, `PA`, `PG`, `PY`, `PE`, `PH`, `PN`, `PL`, `PT`, `PR`, `QA`, `RO`, `RU`, `RW`, `WS`, `SM`, `ST`, `SA`, `SN`, `RS`, `SC`, `SL`, `SG`, `SK`, `SI`, `SB`, `SO`, `ZA`, `KR`, `ES`, `LK`, `SH`, `VC`, `SR`, `SE`, `CH`, `TW`, `TJ`, `TZ`, `TH`, `TL`, `TG`, `TO`, `TT`, `TN`, `TR`, `TM`, `VI`, `UG`, `UA`, `AE`, `GB`, `US`, `UY`, `UZ`, `VU`, `VE`, `VN`, `ZM`, `ZW`.
        language: The parameter specifies the language to use.

Available languages: `en`, `de`, `es`, `es-419`, `fr`, `hr`, `it`, `nl`, `pl`, `pt-BR`, `pt-PT`, `vi`, `tr`, `ru`, `ar`, `th`, `ko`, `zh-CN`, `zh-TW`, `ja`, `ach`, `af`, `ak`, `ig`, `az`, `ban`, `ceb`, `xx-bork`, `bs`, `br`, `ca`, `cs`, `sn`, `co`, `cy`, `da`, `yo`, `et`, `xx-elmer`, `eo`, `eu`, `ee`, `tl`, `fil`, `fo`, `fy`, `gaa`, `ga`, `gd`, `gl`, `gn`, `xx-hacker`, `ht`, `ha`, `haw`, `bem`, `rn`, `id`, `ia`, `xh`, `zu`, `is`, `jw`, `rw`, `sw`, `tlh`, `kg`, `mfe`, `kri`, `la`, `lv`, `to`, `lt`, `ln`, `loz`, `lua`, `lg`, `hu`, `mg`, `mt`, `mi`, `ms`, `pcm`, `no`, `nso`, `ny`, `nn`, `uz`, `oc`, `om`, `xx-pirate`, `ro`, `rm`, `qu`, `nyn`, `crs`, `sq`, `sk`, `sl`, `so`, `st`, `sr-ME`, `sr-Latn`, `su`, `fi`, `sv`, `tn`, `tum`, `tk`, `tw`, `wo`, `el`, `be`, `bg`, `ky`, `kk`, `mk`, `mn`, `sr`, `tt`, `tg`, `uk`, `ka`, `hy`, `yi`, `iw`, `ug`, `ur`, `ps`, `sd`, `fa`, `ckb`, `ti`, `am`, `ne`, `mr`, `hi`, `bn`, `pa`, `gu`, `or`, `ta`, `te`, `kn`, `ml`, `si`, `lo`, `my`, `km`, `chr`.
        skipplaces: Skip first N places, where N should be multiple to 20 (e.g. 0, 20, 40). It's commonly used in pagination.
        limit: The parameter specifies the limit of organizations to take from one query search.

If using more than 20 places per request each additional 20 places will be counted as one search.

There are no more than 500 places per one query search on Maps. For densely populated areas you might want to split your query into subqueries in order to get all the places inside. (e.g., `restaurants, Brooklyn 11211`, `restaurants, Brooklyn 11215`).
        
    """
    url = f"https://local-businesses-by-outscraper.p.rapidapi.com/maps/search-v2"
    querystring = {'query': query, }
    if fields:
        querystring['fields'] = fields
    if region:
        querystring['region'] = region
    if language:
        querystring['language'] = language
    if skipplaces:
        querystring['skipPlaces'] = skipplaces
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-businesses-by-outscraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_places_reviews(query: str, language: str='en', fields: str=None, limit: int=1, reviewsquery: str='amazing', reviewslimit: int=10, cutoffrating: int=None, sort: str='most_relevant', ignoreempty: bool=None, region: str='US', start: int=None, cutoff: str=None, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Places Maps reviews from places when using search queries (e.g., `restaurants, Manhattan, NY, USA`) or from a single place when using IDs or names (e.g., `NoMad Restaurant, NY, USA`, `0x886916e8bc273979:0x5141fcb11460b226`, `ChIJrZhup4lZwokRUr_5sLoFlDw`). In addition to the reviews, it returns places information.
		
		In case no reviews were found by your search criteria, your search request will consume the usage of one request.
		
		This endpoint is optimized for fast responses and can be used as a real-time API. Set the `reviewsLimit` parameter to 10 to achieve the maximum response speed."
    query: The parameter defines the query you want to search. You can use anything that you would use on a regular Maps site. Additionally, you can use `feature_id`, `place_id`, or `CID`. The example of valid queries: `Real estate agency, Rome, Italy`, `The NoMad Restaurant, NY, USA`, `restaurants, Brooklyn 11203`, `0x886916e8bc273979:0x5141fcb11460b226`, `ChIJrZhup4lZwokRUr_5sLoFlDw`, etc.

It supports batching by sending arrays with up to 25 queries (e.g., `query=text1&query=text2&query=text3`). It allows multiple queries to be sent in one request and save on network latency time.

        language: The parameter specifies the language to use.

Available languages: `en`, `de`, `es`, `es-419`, `fr`, `hr`, `it`, `nl`, `pl`, `pt-BR`, `pt-PT`, `vi`, `tr`, `ru`, `ar`, `th`, `ko`, `zh-CN`, `zh-TW`, `ja`, `ach`, `af`, `ak`, `ig`, `az`, `ban`, `ceb`, `xx-bork`, `bs`, `br`, `ca`, `cs`, `sn`, `co`, `cy`, `da`, `yo`, `et`, `xx-elmer`, `eo`, `eu`, `ee`, `tl`, `fil`, `fo`, `fy`, `gaa`, `ga`, `gd`, `gl`, `gn`, `xx-hacker`, `ht`, `ha`, `haw`, `bem`, `rn`, `id`, `ia`, `xh`, `zu`, `is`, `jw`, `rw`, `sw`, `tlh`, `kg`, `mfe`, `kri`, `la`, `lv`, `to`, `lt`, `ln`, `loz`, `lua`, `lg`, `hu`, `mg`, `mt`, `mi`, `ms`, `pcm`, `no`, `nso`, `ny`, `nn`, `uz`, `oc`, `om`, `xx-pirate`, `ro`, `rm`, `qu`, `nyn`, `crs`, `sq`, `sk`, `sl`, `so`, `st`, `sr-ME`, `sr-Latn`, `su`, `fi`, `sv`, `tn`, `tum`, `tk`, `tw`, `wo`, `el`, `be`, `bg`, `ky`, `kk`, `mk`, `mn`, `sr`, `tt`, `tg`, `uk`, `ka`, `hy`, `yi`, `iw`, `ug`, `ur`, `ps`, `sd`, `fa`, `ckb`, `ti`, `am`, `ne`, `mr`, `hi`, `bn`, `pa`, `gu`, `or`, `ta`, `te`, `kn`, `ml`, `si`, `lo`, `my`, `km`, `chr`.
        fields: The parameter defines which fields you want to include with each item returned in the response. By default, it returns all fields. Use `&fields=query,name` to return only the specific ones.
        limit: The parameter specifies the limit of places to take from one query search.
        reviewsquery: The parameter specifies the query to search among the reviews (e.g. `wow`, `amazing`, `horrible place`).
        reviewslimit: The parameter specifies the limit of reviews to get from one place (0 - unlimited).

If using more than 20 reviews per request each additional 10 reviews will be counted as one search.
        cutoffrating: The parameter specifies the maximum for `lowest_rating` sorting or the minimum for `highest_rating` sorting rating for reviews.

Using the cutoffRating requires sorting to be set to `lowest_rating` or `highest_rating.`

        sort: The parameter specifies one of the sorting types: `most_relevant`, `newest`, `highest_rating`, `lowest_rating`.

        ignoreempty: The parameter specifies whether to ignore reviews without text or not.
        region: The parameter specifies the country to use.

Available countries: `AF`, `AL`, `DZ`, `AS`, `AD`, `AO`, `AI`, `AG`, `AR`, `AM`, `AU`, `AT`, `AZ`, `BS`, `BH`, `BD`, `BY`, `BE`, `BZ`, `BJ`, `BT`, `BO`, `BA`, `BW`, `BR`, `VG`, `BN`, `BG`, `BF`, `BI`, `KH`, `CM`, `CA`, `CV`, `CF`, `TD`, `CL`, `CN`, `CO`, `CG`, `CD`, `CK`, `CR`, `CI`, `HR`, `CU`, `CY`, `CZ`, `DK`, `DJ`, `DM`, `DO`, `EC`, `EG`, `SV`, `EE`, `ET`, `FJ`, `FI`, `FR`, `GA`, `GM`, `GE`, `DE`, `GH`, `GI`, `GR`, `GL`, `GT`, `GG`, `GY`, `HT`, `HN`, `HK`, `HU`, `IS`, `IN`, `ID`, `IQ`, `IE`, `IM`, `IL`, `IT`, `JM`, `JP`, `JE`, `JO`, `KZ`, `KE`, `KI`, `KW`, `KG`, `LA`, `LV`, `LB`, `LS`, `LY`, `LI`, `LT`, `LU`, `MG`, `MW`, `MY`, `MV`, `ML`, `MT`, `MU`, `MX`, `FM`, `MD`, `MN`, `ME`, `MS`, `MA`, `MQ`, `MZ`, `MM`, `NA`, `NR`, `NP`, `NL`, `NZ`, `NI`, `NE`, `NG`, `NU`, `MK`, `NO`, `OM`, `PK`, `PS`, `PA`, `PG`, `PY`, `PE`, `PH`, `PN`, `PL`, `PT`, `PR`, `QA`, `RO`, `RU`, `RW`, `WS`, `SM`, `ST`, `SA`, `SN`, `RS`, `SC`, `SL`, `SG`, `SK`, `SI`, `SB`, `SO`, `ZA`, `KR`, `ES`, `LK`, `SH`, `VC`, `SR`, `SE`, `CH`, `TW`, `TJ`, `TZ`, `TH`, `TL`, `TG`, `TO`, `TT`, `TN`, `TR`, `TM`, `VI`, `UG`, `UA`, `AE`, `GB`, `US`, `UY`, `UZ`, `VU`, `VE`, `VN`, `ZM`, `ZW`.
        start: The parameter specifies the start timestamp value for reviews. The current timestamp is used when the value is not provided.

Using the start parameter overwrites the `sort` parameter to `newest`. Therefore, the latest reviews will be at the beginning.

        cutoff: The parameter specifies the oldest timestamp value for reviews.

Using the cutoff parameter overwrites the `sort` parameter to `newest`. Therefore, the latest reviews will be at the beginning.
        skip: The parameter specifies the number of items to skip. It's commonly used in pagination.
        
    """
    url = f"https://local-businesses-by-outscraper.p.rapidapi.com/maps/reviews-v3"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    if fields:
        querystring['fields'] = fields
    if limit:
        querystring['limit'] = limit
    if reviewsquery:
        querystring['reviewsQuery'] = reviewsquery
    if reviewslimit:
        querystring['reviewsLimit'] = reviewslimit
    if cutoffrating:
        querystring['cutoffRating'] = cutoffrating
    if sort:
        querystring['sort'] = sort
    if ignoreempty:
        querystring['ignoreEmpty'] = ignoreempty
    if region:
        querystring['region'] = region
    if start:
        querystring['start'] = start
    if cutoff:
        querystring['cutoff'] = cutoff
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "local-businesses-by-outscraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

