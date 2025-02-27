import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def class_2_0(model: str, abstract: str='The 85th Academy Awards review', expand_hierarchy: str=None, of: str='json', txt: str='The 85th Academy Awards ceremony took place February 24, 2013.', url: str='null', title: str='null', verbose: str='n', categories: str='0800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Automatic classification of multilingual texts"
    model: Classification model to use. It will define into which categories the text may be classified. Possilbe values are: IPTC_es, IPTC_en, IPTC_ca, IPTC_pt, IPTC_it, IPTC_fr, EUROVOC_es_ca, BusinessRep_es, BusinessRepShort_es
        of: Output formatl, xml or json
        txt: Input text. It can be plain text, HTML or XML, always using UTF-8 encoding. (Required if 'doc' and 'url' are empty)
        url: URL with the content to classify. Currently only non-authenticated HTTP and FTP are supported. The content types supported for URL contents can be found at https://textalytics.com/core/supported-formats. (Required if 'txt' and 'doc' are empty)
        title: Descriptive title of the content. It is an optional field, and it can be plain text, HTML or XML, always using UTF-8 encoding. The terms relevant for the classification process found in the title will have more influence in the classification than if they were in the text.
        verbose: Verbose mode. Shows additional information about the classification.
        categories: List of prefixes of the code of the categories to which the classification is limited. Each value will be separated by '|'. All the categories that do not start with any of the prefixes specified in the list will not be taken account in the classification. For example, if only a clasification within the human interest category, the prefix used would be 0800.
        
    """
    url = f"https://text-classification.p.rapidapi.com/class-2.0"
    querystring = {'model': model, }
    if abstract:
        querystring['abstract'] = abstract
    if expand_hierarchy:
        querystring['expand_hierarchy'] = expand_hierarchy
    if of:
        querystring['of'] = of
    if txt:
        querystring['txt'] = txt
    if url:
        querystring['url'] = url
    if title:
        querystring['title'] = title
    if verbose:
        querystring['verbose'] = verbose
    if categories:
        querystring['categories'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-classification.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

