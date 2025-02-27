import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def suggest(locale: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions"
    locale: `ar_AE` `cs_CZ` `da_DK` `de_AT` `de_BE` `de_CH` `de_DE` `el_GR` `en_AS` `en_AU` `en_CA` `en_CN` `en_GB` `en_HK` `en_ID` `en_IE` `en_IL` `en_IN` `en_JP` `en_KR` `en_LA` `en_MX` `en_MY` `en_NZ` `en_PH` `en_SG` `en_TH` `en_TW` `en_US` `en_VN` `en_ZA` `es_AR` `es_BO` `es_BZ` `es_CL` `es_CO` `es_CR` `es_EC` `es_ES` `es_GT` `es_GY` `es_HN` `es_MX` `es_NI` `es_PA` `es_PE` `es_PY` `es_SV` `es_US` `es_UY` `es_VE` `et_EE` `fi_FI` `fr_BE` `fr_CA` `fr_CH` `fr_FR` `fr_GF` `hr_HR` `hu_HU` `in_ID` `is_IS` `it_CH` `it_IT` `iw_IL` `ja_JP` `ko_KR` `lt_LT` `lv_LV` `ms_MY` `nl_BE` `nl_NL` `nl_SR` `no_NO` `pl_PL` `pt_BR` `pt_PT` `ru_RU` `sk_SK` `sv_SE` `th_TH` `tr_TR` `uk_UA` `vi_VN` `zh_CN` `zh_HK` `zh_TW`
        
    """
    url = f"https://hotels-com-free.p.rapidapi.com/suggest/v1.7/json"
    querystring = {'locale': locale, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-free.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

