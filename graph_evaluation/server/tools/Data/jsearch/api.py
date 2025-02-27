import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def estimated_salary(location: str, job_title: str, radius: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get estimated salaries for a jobs around a location."
    location: Location in which to get salary estimation.
        job_title: Job title for which to get salary estimation.
        radius: Search radius in km (measured from location).
Default: `200`.
        
    """
    url = f"https://jsearch.p.rapidapi.com/estimated-salary"
    querystring = {'location': location, 'job_title': job_title, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def job_details(job_id: str, extended_publisher_details: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all job details, including additional application options / links, employer reviews and estimated salaries for similar jobs."
    job_id: Job Id of the job for which to get details. Batching of up to 20 Job Ids is supported by separating multiple Job Ids by comma (,).

Note that each Job Id in a batch request is counted as a request for quota calculation.
        extended_publisher_details: Return additional publisher details such as website url and favicon.
        
    """
    url = f"https://jsearch.p.rapidapi.com/job-details"
    querystring = {'job_id': job_id, }
    if extended_publisher_details:
        querystring['extended_publisher_details'] = extended_publisher_details
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_filters(query: str, language: str=None, country: str=None, categories: str=None, company_types: str=None, job_titles: str=None, job_requirements: str=None, radius: int=None, employers: str=None, remote_jobs_only: bool=None, employment_types: str=None, date_posted: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts all **Search** endpoint parameters (except for `page` and `num_pages`) and returns the relevant filters and their estimated result counts for later use in search or for analytics."
    query: Free-form jobs search query. It is highly recommended to include job title and location as part of the query, see query examples below.

**Query examples**
- *web development in chicago*
- *marketing manager in new york via linkedin*
- *developer in germany 60306*
        language: [EXPERIMENTAL]

Set the language of the results. In case set, Google for Jobs might prefer jobs that were posted in the specified language.

Allowed values: 2-letter language code, see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
Default: `en`.
        country: [EXPERIMENTAL]

The country / region from which to make the query.

Allowed values: 2-letter country code, see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
Default: `us`.
        categories: **[Deprecated]** Categories/industries filter - specified as a comma (,) separated list of `categories` filter values (i.e. filter *value* field) as returned by a previous call to this endpoint.

**Example**: *categories=R0MxODpNYW5hZ2VtZW50,R0MwNTpBcnRGYXNoaW9uRGVzaWdu*
        company_types: Company types filter - specified as a comma (,) separated list of `company_types` filter values (i.e. filter *value* field) as returned by a previous call to this endpoint.

**Example**: *company_types= L2J1c2luZXNzL25haWNzMjAwNy81MjpGaW5hbmNl,L2J1c2luZXNzL25haWNzMjAwNy81MTpJbmZvcm1hdGlvbg==*
        job_titles: Job title filter - specified as a comma (,) separated list of `job_titles` filter values (i.e. filter *value* field) as returned by a previous call to this endpoint.

**Example**: *job_titles=c2VuaW9y,YXNzb2NpYXRl*
        job_requirements: Find jobs with specific requirements, specified as a comma delimited list of the following values: `under_3_years_experience`, `more_than_3_years_experience`, `no_experience`, `no_degree`.
        radius: Return jobs within a certain distance from location as specified as part of the query (in km).
        employers: Employers filter - specified as a comma (,) separated list of `employers` filter values (i.e. filter *value* field) as returned by a previous call to this endpoint.

**Example**: *employers= L2cvMTFoMTV4eHhydDpJbmZpbml0eSBDb25zdWx0aW5n,L2cvMTFmMDEzOXIxbjpDeWJlckNvZGVycw==*
        remote_jobs_only: Find remote jobs only (work from home).
Default: `false`.
        employment_types: Find jobs of particular employment types, specified as a comma delimited list of the following values: `FULLTIME`, `CONTRACTOR`, `PARTTIME`, `INTERN`.
        date_posted: Find jobs posted within the time you specify.
Possible values: `all`, `today`, `3days`, `week`,`month`.
Default: `all`.
        
    """
    url = f"https://jsearch.p.rapidapi.com/search-filters"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    if categories:
        querystring['categories'] = categories
    if company_types:
        querystring['company_types'] = company_types
    if job_titles:
        querystring['job_titles'] = job_titles
    if job_requirements:
        querystring['job_requirements'] = job_requirements
    if radius:
        querystring['radius'] = radius
    if employers:
        querystring['employers'] = employers
    if remote_jobs_only:
        querystring['remote_jobs_only'] = remote_jobs_only
    if employment_types:
        querystring['employment_types'] = employment_types
    if date_posted:
        querystring['date_posted'] = date_posted
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, exclude_job_publishers: str=None, categories: str=None, radius: int=None, language: str=None, country: str=None, employer: str=None, job_requirements: str=None, remote_jobs_only: bool=None, job_titles: str=None, company_types: str=None, num_pages: str='1', date_posted: str=None, page: int=1, employment_types: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for jobs posted on job sites across the web on the largest job aggregate in the world - Google for Jobs. Extensive filtering support and most options available on Google for Jobs."
    query: Free-form jobs search query. It is highly recommended to include job title and location as part of the query, see query examples below.

**Query examples**
- *web development in chicago*
- *marketing manager in new york via linkedin*
- *developer in germany 60306*
        exclude_job_publishers: Exclude jobs published by specific publishers, specified as a comma (,) separated list of publishers to exclude.

**Example**: 
*exclude_job_publishers=BeeBe,Dice*
        categories: **[Deprecated]** Find jobs in specific categories/industries - specified as a comma (,) separated list of `categories` filter values (i.e. filter *value* field) as returned by the **Search Filters** endpoint.

**Example**: *categories=R0MxODpNYW5hZ2VtZW50,R0MwNTpBcnRGYXNoaW9uRGVzaWdu*
        radius: Return jobs within a certain distance from location as specified as part of the query (in km).
        language: [EXPERIMENTAL]

Set the language of the results. In case set, Google for Jobs might prefer jobs that were posted in the specified language.

Allowed values: 2-letter language code, see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
Default: `en`.
        country: [EXPERIMENTAL]

The country / region from which to make the query.

Allowed values: 2-letter country code, see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
Default: `us`.
        employer: Find jobs posted by specific employers - specified as a comma (,) separated list of `employer` filter values (i.e. filter *value* field) as returned by the **Search Filters** endpoint.

**Example**: *employers= L2cvMTFoMTV4eHhydDpJbmZpbml0eSBDb25zdWx0aW5n,L2cvMTFmMDEzOXIxbjpDeWJlckNvZGVycw==*
        job_requirements: Find jobs with specific requirements, specified as a comma delimited list of the following values: `under_3_years_experience`, `more_than_3_years_experience`, `no_experience`, `no_degree`.
        remote_jobs_only: Find remote jobs only (work from home).
Default: `false`.
        job_titles: Find jobs with specific job titles - specified as a comma (,) separated list of `job_titles` filter values (i.e. filter *value* field) as returned by the **Search Filters** endpoint.

**Example**: *job_titles=c2VuaW9y,YXNzb2NpYXRl*
        company_types: Find jobs posted by companies of certain types - specified as a comma (,) separated list of `company_types` filter values (i.e. filter *value* field) as returned by the **Search Filters** endpoint.

**Example**: *company_types= L2J1c2luZXNzL25haWNzMjAwNy81MjpGaW5hbmNl,L2J1c2luZXNzL25haWNzMjAwNy81MTpJbmZvcm1hdGlvbg==*
        num_pages: Number of pages to return, starting from `page`.
Allowed values: `1-20`.
Default: `1`.

 **Note**: requests for more than one page and up to 10 pages are charged x2 and requests for more than 10 pages are charged 3x.
        date_posted: Find jobs posted within the time you specify.
Allowed values: `all`, `today`, `3days`, `week`,`month`.
Default: `all`.
        page: Page to return (each page includes up to 10 results).
Allowed values: `1-100`.
Default: `1`.
        employment_types: Find jobs of particular employment types, specified as a comma delimited list of the following values: `FULLTIME`, `CONTRACTOR`, `PARTTIME`, `INTERN`.
        
    """
    url = f"https://jsearch.p.rapidapi.com/search"
    querystring = {'query': query, }
    if exclude_job_publishers:
        querystring['exclude_job_publishers'] = exclude_job_publishers
    if categories:
        querystring['categories'] = categories
    if radius:
        querystring['radius'] = radius
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    if employer:
        querystring['employer'] = employer
    if job_requirements:
        querystring['job_requirements'] = job_requirements
    if remote_jobs_only:
        querystring['remote_jobs_only'] = remote_jobs_only
    if job_titles:
        querystring['job_titles'] = job_titles
    if company_types:
        querystring['company_types'] = company_types
    if num_pages:
        querystring['num_pages'] = num_pages
    if date_posted:
        querystring['date_posted'] = date_posted
    if page:
        querystring['page'] = page
    if employment_types:
        querystring['employment_types'] = employment_types
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

