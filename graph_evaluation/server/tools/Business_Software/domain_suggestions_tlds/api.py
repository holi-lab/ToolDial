import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_v1_suggestions(str: str, all_in_one_list: bool=None, tld: str=None, num_tlds: int=30, use_generic_tlds: bool=None, tlds_use: str='[]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "100s of domain name suggestions + related TLDs. More options (Query Params / POST data) will be added as soon as they are available and tested. Please check back soon. Also see our Availability API. More options and features coming soon. All changes will be backward compatible. Read more about this API and future improvements at https://besta.domains/api"
    str: The domain you are searching for, with or without the TLD. Spaces are allowed, and encouraged! Keeping spaces helps us decide where to break the words. But we usually break the word just fine even without spaces. Try it a few times! Please let us know how it goes.

NOTE: When testing on RapidAPI.com, wait a second after typing before hitting \"Enter\". RapidAPI input fields have a lag, and may cut off your text if submitted too fast.
        all_in_one_list: (coming soon) Currently, suggestions are returned in several lists - grouped by type of manipulation. Set this to true, to return all suggestions mixed into one list. We're still deciding the best strategy to mix and sort the different types of suggestions. Please tell us your thoughts: https://besta.domains/contact
        tld: (optional) (default=\"com\") The TLD you wish to use. No spaces, no quotes, NO dot. Must be a valid TLD. We do not check validity of TLD you send. If you specify a TLD as part of the domain, then this value will be ignored.
        num_tlds: (optional) (default=30) How many related TLDs should we find, including the ones you specified and \".com\"? We return relevant TLDs in data.tlds. However, we also use them in data.domains. So, if you specify a small number, you will also have fewer name suggestions. However, they may be more relevant, because you'd be using only the most relevant TLDs.
        use_generic_tlds: (coming soon) (optional) (default=true) Besides related relevant TLDs which we find according to similarity to keywords - we also mix in some generic TLDs to create name suggestions. These are \"com\", \"net\", \"info\", \"biz\", etc. Set this to falce to disable this.
        tlds_use: (optional) (experimental) (default=[\"com\"]) Use multiple TLDs, ordered by importance. First TLD will be considered much more important than others. We will also find other relevant TLDs whether you send this value or not, as part of our suggestions. However, this list will be used first. Then, our relevant TLD suggestions will be added after this.

For your convenience, you may send either: 1) JSON Array 2) or \"comma, separated, string\" with no quotes. TLD must be included. Spaces are allowed. If you specify a TLD as part of the \"domain\", or specify a \"tld\", then this value will be ignored.
        
    """
    url = f"https://domain-suggestions-tlds.p.rapidapi.com/v1/suggestions"
    querystring = {'str': str, }
    if all_in_one_list:
        querystring['all_in_one_list'] = all_in_one_list
    if tld:
        querystring['tld'] = tld
    if num_tlds:
        querystring['num_tlds'] = num_tlds
    if use_generic_tlds:
        querystring['use_generic_tlds'] = use_generic_tlds
    if tlds_use:
        querystring['tlds_use'] = tlds_use
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-suggestions-tlds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

