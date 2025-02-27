import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_company_details_from_crunchbase_new(crunchbase: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all public details of a company on Crunchbase - including funding rounds. **2 credits per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-company-details-from-crunchbase"
    querystring = {'crunchbase': crunchbase, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_ads_count(company_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get number of ads the company posted on LinkedIn. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-company-ads-count"
    querystring = {'company_id': company_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_extra_profile_data(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more profile’s data fields like languages, top skills, certifications, publications, patents, awards"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-extra-profile-data"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_by_id(company_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company’s LinkedIn internal ID, the API will return valuable data points in JSON format. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-company-by-id"
    querystring = {'company_id': company_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_job_details(job_url: str, include_skills: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape the full job details, including the company basic information. **1 credit per call.**"
    include_skills: Including skills will cost 1 more credit
        
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-job-details"
    querystring = {'job_url': job_url, }
    if include_skills:
        querystring['include_skills'] = include_skills
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_by_url(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company’s LinkedIn URL, the API will return valuable data points in JSON format. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-company-by-linkedinurl"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recommendation_received(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get profile’s recommendations (received). **1 credit per call**."
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-recommendations-received"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recommendation_given(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get profile’s recommendations (given). **1 credit per call**."
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-recommendations-given"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_jobs(industry_code: str='4,5', company_id: int=None, sort_by: str='most_relevant', date_posted: str='any_time', salary: str=None, onsite_remote: str=None, start: int=0, experience_level: str=None, function_id: str='it,sale', geo_code: int=103644278, title_id: str=None, keywords: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search jobs posted on LinkedIn. This endpoint is useful for scraping job openings of a specific company on LinkedIn. 
		
		To scrape all results from each search, change the param *start* from 0 to 25, 50, ... until you see less than 25 results returned.
		
		**2 credits per call.**"
    industry_code: You can find all valid industry codes from [this page]( https://learn.microsoft.com/en-us/linkedin/shared/references/reference-tables/industry-codes).
        sort_by: Possible values:  most_relevant, most_recent
        date_posted: Possible values: any_time,  past_month, past_week, past_24_hours

        salary: Possible values: 40k+, 60k+, 80k+, 100k+, 120k+, 140k+, 160k+, 180k+, 200k+
        onsite_remote: Possible values: on-site, remote, hybrid
        start: Should be one of: 0, 25, 50, 75, etc.
        experience_level: Possible values: internship, associate, director, entry level, mid-senior level, executive
        function_id: Please follow [this instruction](https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data/tutorials/how-to-find-function_id-on-linkedin%3F) to get the function_id of your choice.
        geo_code: Use this param to target jobs in specific region/country. To search worldwide, use 92000000.
To find other geo codes, please follow this [link](https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data/tutorials/how-to-find-a-geo_code-(geoid)-on-linkedin%3F)
        title_id: To find title_id by title, please follow this [link](https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data/tutorials/how-to-find-a-title_id-on-linkedin%3F)
        
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/search-jobs"
    querystring = {}
    if industry_code:
        querystring['industry_code'] = industry_code
    if company_id:
        querystring['company_id'] = company_id
    if sort_by:
        querystring['sort_by'] = sort_by
    if date_posted:
        querystring['date_posted'] = date_posted
    if salary:
        querystring['salary'] = salary
    if onsite_remote:
        querystring['onsite_remote'] = onsite_remote
    if start:
        querystring['start'] = start
    if experience_level:
        querystring['experience_level'] = experience_level
    if function_id:
        querystring['function_id'] = function_id
    if geo_code:
        querystring['geo_code'] = geo_code
    if title_id:
        querystring['title_id'] = title_id
    if keywords:
        querystring['keywords'] = keywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_s_posts(linkedin_url: str, pagination_token: str=None, start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts from a LinkedIn company page.  Pagination is supported to fetch all posts. **2 credits per call.**"
    pagination_token: Required when fetching the next result page. Please use the token from the result of your previous call.
        start: Use this param to fetch posts of the next result page: 0 for page 1, 50 for page 2, etc.
        
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-company-posts"
    querystring = {'linkedin_url': linkedin_url, }
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_by_domain(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find a company on LinkedIn using its web domain. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-company-by-domain"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_personal_profile(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full profile data, including experience & education history, skillset and company related details. Accept all type of profile urls. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-linkedin-profile"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pesonal_profile_by_sales_nav_url(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full profile data, including experience & education history, skillset and company related details. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-linkedin-profile-by-salesnavurl"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_profile_s_posts(linkedin_url: str, start: int=None, pagination_token: str=None, type: str='posts', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get posts of a person based on profile url. Pagination is supported to get all posts. **2 credits per call.**"
    start: Use this param to fetch posts of the next result page: 0 for page 1, 50 for page 2, etc.
        pagination_token: Required when fetching the next result page. Please use the token from the result of your previous call.
        type: Possible values: 

- posts: to scrape posts from tab Posts -- posts or posts reshared by the person

- comments: to scrape posts from tab Comments -- posts the person commented

- reactions: to scrape posts from tab Reactions -- posts the person reacted


        
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-profile-posts"
    querystring = {'linkedin_url': linkedin_url, }
    if start:
        querystring['start'] = start
    if pagination_token:
        querystring['pagination_token'] = pagination_token
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_jobs_count(company_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get number of opening jobs the company posted on LinkedIn. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-company-jobs-count"
    querystring = {'company_id': company_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_profile_recent_activity_time(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the time of the latest activity. **2 credits per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-profile-recent-activity-time"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_open_to_work_status(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a LinkedIn profile URL, the API will let you know if that profile is “open to work” or not. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-opentowork-status"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_profile_pdf_cv(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the CV of a LinkedIn profile in PDF format. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-profile-pdf-cv"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_search_status(request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the status of your search using the request_id given in step 1."
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/check-search-status"
    querystring = {'request_id': request_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_search_results(page: str, request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search results. Please make sure the search is "done" before calling this endpoint."
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-search-results"
    querystring = {'page': page, 'request_id': request_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_open_profile_status(linkedin_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a LinkedIn profile URL, the API will let you know if that profile is “open profile” or not. **1 credit per call.**"
    
    """
    url = f"https://fresh-linkedin-profile-data.p.rapidapi.com/get-open-profile-status"
    querystring = {'linkedin_url': linkedin_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fresh-linkedin-profile-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

