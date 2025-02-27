import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_comments_ccfc31c2_0e94_4db6_9956_382d4c9efe6e(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scope under which the request is made; determines fields present in response."
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/comments"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_statuses_c0a5199c_546e_4cab_8f8f_73c99e62c988(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scope under which the request is made; determines fields present in response."
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/statuses"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tags_c0b25f9e_c759_46ae_8d6f_14e6404e49a3(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List Tags"
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/tags"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_id_revisions_16a458a3_dbc6_4bec_aee5_93db717a117a(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get post revisions"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/posts/{is_id}/revisions"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_id_adb0e84c_5f56_4e25_8aba_9042e0c3f90f(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Media"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/media/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_8125bc02_6037_4c8c_8102_2297920726f0(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scope under which the request is made; determines fields present in response."
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/media"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_statuses_id_1853ae07_ba42_43d0_8a00_00f562d98e48(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Status"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/statuses/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_id_9d548709_72a0_48c0_a0fa_57c41026a2ef(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Post"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/posts/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pages_id_04cce496_738c_44ef_9faf_4d42e99c642b(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Page"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/pages/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories_36a9b66a_641f_4969_a849_36119ad674a3(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List categories"
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/categories"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_types_852347ee_35c5_44fb_8548_1bee078d2350(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scope under which the request is made; determines fields present in response."
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/types"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories_id_411760b4_8ab7_4a73_a04a_6e0cb2e2c00c(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Category"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/categories/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_types_id_523e7700_867d_45a5_8362_f7d4818f6bc9(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Type"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/types/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comments_id_68712d36_48f7_4050_9c6a_a1e8491a8477(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Comment"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/comments/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_taxonomies_6b249ef7_e2c2_4378_b40a_42146d250423(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List Taxonomy"
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/taxonomies"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users_68a613f6_d503_4942_9c11_e04e59427ec1(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List Tags"
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/users"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_users_id_3449a5e3_22f0_4216_bb83_c737b2cef1b0(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single User"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/users/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_taxonomies_id_b2a5d5a0_3692_43aa_ae85_624855b4cde8(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Taxonomy"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/taxonomies/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_2504660f_8ad6_4815_86e0_312cc593fe26(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scope under which the request is made; determines fields present in response."
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/posts"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_posts_id_revisions_revisionid_2156a9ee_f3cf_44f0_be60_37fbeb2d9b95(revisionid: str, is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get single post revisions"
    revisionid: Id of revision
        id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/posts/{is_id}/revisions/{revisionid}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tags_id_b45ea222_6d54_4156_b9e6_5630c4edffe7(is_id: str, context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Tag"
    id: Id of object
        context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/tags/{is_id}"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pages_fb3c1d3c_13c4_4cb5_9a4d_5dbc5e195d42(context: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scope under which the request is made; determines fields present in response."
    context: Scope under which the request is made; determines fields present in response.
        
    """
    url = f"https://the-south-asian-express.p.rapidapi.com/pages"
    querystring = {}
    if context:
        querystring['context'] = context
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-south-asian-express.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

