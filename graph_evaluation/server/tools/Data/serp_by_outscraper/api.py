import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, region: str='US', language: str='en', pagesperquery: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns search results based on a given search query (or many queries). This endpoint is optimized for fast responses and can be used as a real-time API.
		
		[API Docs](https://app.outscraper.com/api-docs)"
    query: Queries to search (e.g., `bitcoin`, `37th president of USA`).

It supports batching by sending arrays with up to 25 queries (e.g., `query=text1&query=text2&query=text3`). It allows multiple queries to be sent in one request and to save on network latency time (each query will be counted as a different request in billing).

        region: The parameter specifies the country to use.

Available countries: `AF`, `AL`, `DZ`, `AS`, `AD`, `AO`, `AI`, `AG`, `AR`, `AM`, `AU`, `AT`, `AZ`, `BS`, `BH`, `BD`, `BY`, `BE`, `BZ`, `BJ`, `BT`, `BO`, `BA`, `BW`, `BR`, `VG`, `BN`, `BG`, `BF`, `BI`, `KH`, `CM`, `CA`, `CV`, `CF`, `TD`, `CL`, `CN`, `CO`, `CG`, `CD`, `CK`, `CR`, `CI`, `HR`, `CU`, `CY`, `CZ`, `DK`, `DJ`, `DM`, `DO`, `EC`, `EG`, `SV`, `EE`, `ET`, `FJ`, `FI`, `FR`, `GA`, `GM`, `GE`, `DE`, `GH`, `GI`, `GR`, `GL`, `GT`, `GG`, `GY`, `HT`, `HN`, `HK`, `HU`, `IS`, `IN`, `ID`, `IQ`, `IE`, `IM`, `IL`, `IT`, `JM`, `JP`, `JE`, `JO`, `KZ`, `KE`, `KI`, `KW`, `KG`, `LA`, `LV`, `LB`, `LS`, `LY`, `LI`, `LT`, `LU`, `MG`, `MW`, `MY`, `MV`, `ML`, `MT`, `MU`, `MX`, `FM`, `MD`, `MN`, `ME`, `MS`, `MA`, `MQ`, `MZ`, `MM`, `NA`, `NR`, `NP`, `NL`, `NZ`, `NI`, `NE`, `NG`, `NU`, `MK`, `NO`, `OM`, `PK`, `PS`, `PA`, `PG`, `PY`, `PE`, `PH`, `PN`, `PL`, `PT`, `PR`, `QA`, `RO`, `RU`, `RW`, `WS`, `SM`, `ST`, `SA`, `SN`, `RS`, `SC`, `SL`, `SG`, `SK`, `SI`, `SB`, `SO`, `ZA`, `KR`, `ES`, `LK`, `SH`, `VC`, `SR`, `SE`, `CH`, `TW`, `TJ`, `TZ`, `TH`, `TL`, `TG`, `TO`, `TT`, `TN`, `TR`, `TM`, `VI`, `UG`, `UA`, `AE`, `GB`, `US`, `UY`, `UZ`, `VU`, `VE`, `VN`, `ZM`, `ZW`.
        language: The parameter specifies the language to use.

Available languages: `en`, `de`, `es`, `es-419`, `fr`, `hr`, `it`, `nl`, `pl`, `pt-BR`, `pt-PT`, `vi`, `tr`, `ru`, `ar`, `th`, `ko`, `zh-CN`, `zh-TW`, `ja`, `ach`, `af`, `ak`, `ig`, `az`, `ban`, `ceb`, `xx-bork`, `bs`, `br`, `ca`, `cs`, `sn`, `co`, `cy`, `da`, `yo`, `et`, `xx-elmer`, `eo`, `eu`, `ee`, `tl`, `fil`, `fo`, `fy`, `gaa`, `ga`, `gd`, `gl`, `gn`, `xx-hacker`, `ht`, `ha`, `haw`, `bem`, `rn`, `id`, `ia`, `xh`, `zu`, `is`, `jw`, `rw`, `sw`, `tlh`, `kg`, `mfe`, `kri`, `la`, `lv`, `to`, `lt`, `ln`, `loz`, `lua`, `lg`, `hu`, `mg`, `mt`, `mi`, `ms`, `pcm`, `no`, `nso`, `ny`, `nn`, `uz`, `oc`, `om`, `xx-pirate`, `ro`, `rm`, `qu`, `nyn`, `crs`, `sq`, `sk`, `sl`, `so`, `st`, `sr-ME`, `sr-Latn`, `su`, `fi`, `sv`, `tn`, `tum`, `tk`, `tw`, `wo`, `el`, `be`, `bg`, `ky`, `kk`, `mk`, `mn`, `sr`, `tt`, `tg`, `uk`, `ka`, `hy`, `yi`, `iw`, `ug`, `ur`, `ps`, `sd`, `fa`, `ckb`, `ti`, `am`, `ne`, `mr`, `hi`, `bn`, `pa`, `gu`, `or`, `ta`, `te`, `kn`, `ml`, `si`, `lo`, `my`, `km`, `chr`.
        pagesperquery: The parameter specifies the limit of pages to return from one query.
        
    """
    url = f"https://serp-by-outscraper.p.rapidapi.com/serp-search"
    querystring = {'query': query, }
    if region:
        querystring['region'] = region
    if language:
        querystring['language'] = language
    if pagesperquery:
        querystring['pagesPerQuery'] = pagesperquery
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serp-by-outscraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

