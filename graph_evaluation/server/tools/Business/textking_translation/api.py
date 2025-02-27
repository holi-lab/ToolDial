import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def upload_source_document(projectid: str, jobid: str, documentname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Uploads the source document for a job."
    projectid: The project UUID.
        jobid: The job UUID.
        documentname: A valid file name for the uploaded file.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/project/{projectId}/job/{jobId}/document/{documentName}"
    querystring = {'projectid': projectid, 'jobid': jobid, 'documentname': documentname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_language(code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single language with the given ISO 639 language code."
    code: The ISO 639 language code.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/language/{code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_source_languages(page: int=1, per_page: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of all source languages supported by TEXTKING."
    page: Return search results starting at a given page. Used for paging through more than one page of results.
        per_page: An integer value defining how many entries should be returned. Only values between 1 and 100 (both inclusive) are allowed. If not given, this defaults to 100.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/languages/source"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_topics(per_page: int=100, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of all available topics."
    per_page: An integer value defining how many entries should be returned. Only values between 1 and 100 (both inclusive) are allowed. If not given, this defaults to 100.
        page: Return search results starting at a given page. Used for paging through more than one page of results.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/topics"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_projects(per_page: int=100, page: int=1, state: str='running', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of projects of the authorized customer."
    per_page: An integer value defining how many entries should be returned. Only values between 1 and 100 (both inclusive) are allowed. If not given, this defaults to 100.
        page: Return search results starting at a given page. Used for paging through more than one page of results.
        state: Filter projects by status: prepared, running, finished, canceled
        
    """
    url = f"https://textkingv1.p.rapidapi.com/projects"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_source_document(projectid: str, jobid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads the source document for a job."
    projectid: The project UUID.
        jobid: The job UUID.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/project/{projectId}/job/{jobId}/document"
    querystring = {'projectid': projectid, 'jobid': jobid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_target_languages(page: int=1, per_page: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    page: Return search results starting at a given page. Used for paging through more than one page of results.
        per_page: An integer value defining how many entries should be returned. Only values between 1 and 100 (both inclusive) are allowed. If not given, this defaults to 100.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/languages/target"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_price_info_for_word_count(words: int, source_language: str, target_language: str, topic: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns price information for a specific number of words for all available quality levels."
    words: An integer value defining the number of words to translate.
        source_language: ISO 639 language code of the source language.
        target_language: ISO 639 language code of the target language.
        topic: A topic UUID.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/price-info"
    querystring = {'words': words, 'source_language': source_language, 'target_language': target_language, 'topic': topic, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_project(projectid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single project with the given ID."
    projectid: The project UUID.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/project/{projectId}"
    querystring = {'projectid': projectid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_topic(topicid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single topic area with the given ID."
    topicid: The topic UUID.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/topic/{topicId}"
    querystring = {'topicid': topicid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_job(projectid: str, jobid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single job with the given ID."
    projectid: The project UUID.
        jobid: The job UUID.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/project/{projectId}/job/{jobId}"
    querystring = {'projectid': projectid, 'jobid': jobid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_translated_document(projectid: str, jobid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads the translated document for a job."
    projectid: The project UUID.
        jobid: The job UUID.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/project/{projectId}/job/{jobId}/translation"
    querystring = {'projectid': projectid, 'jobid': jobid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_a_project_s_jobs(per_page: int, projectid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a collection of jobs for a project."
    per_page: An integer value defining how many entries should be returned. Only values between 1 and 100 (both inclusive) are allowed. If not given, this defaults to 100.
        page: Return search results starting at a given page. Used for paging through more than one page of results.
        projectid: The project UUID.
        
    """
    url = f"https://textkingv1.p.rapidapi.com/project/{projectId}/jobs"
    querystring = {'per_page': per_page, 'projectid': projectid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "textkingv1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

