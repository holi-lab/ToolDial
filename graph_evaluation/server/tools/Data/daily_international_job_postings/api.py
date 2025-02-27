import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1_meta_jobs_sources(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a distinct list of sources for a time period.<br /><br /><i>Exemplary Call: `/api/v1/meta/jobs/sources?date=2022-02-01`<br /><br /><i>NOTE: only available for special subscription plans."
    date: The day or month of the time period with the format 'YYYY-MM-DD' (day) or 'YYYY-MM' (month). <br /><br /><i>NOTE: the day must lie 2 days before 'now' as our crawlers need time to completly collect all jobs from the jobboards in different timezones.</i>
        
    """
    url = f"https://daily-international-job-postings.p.rapidapi.com/api/v1/meta/jobs/sources"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-international-job-postings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_meta_jobs_count(date: str, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a count of job postings for a country and a day or month.<br /><br /><i>Exemplary Call: `/api/v1/meta/jobs/count?countryCode=lu&date=2022-02`<br /><br /><i>NOTE: only available for special subscription plans."
    date: The day or month of the time period with the format 'YYYY-MM-DD' (day) or 'YYYY-MM' (month). <br /><br /><i>NOTE: the day must lie 2 days before 'now' as our crawlers need time to completly collect all jobs from the jobboards in different timezones.</i>
        countrycode: A 2-letter countryCode from ISO 3166-1 alpha-2. <br /><br /><i>NOTE: Internally, we lowerCase the countryCode and convert 'gb' into 'uk' and 'el' into 'gr'.</i>
        
    """
    url = f"https://daily-international-job-postings.p.rapidapi.com/api/v1/meta/jobs/count"
    querystring = {'date': date, 'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-international-job-postings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_meta_jobs_countries(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a distinct list of countries for a time period (day or month).<br /><br /><i>Exemplary Call: `/api/v1/meta/jobs/countries?date=2022-02-02`<br /><br /><i>NOTE: only available for special subscription plans."
    date: The day or month of the time period with the format 'YYYY-MM-DD' (day) or 'YYYY-MM' (month). <br /><br /><i>NOTE: the day must lie 2 days before 'now' as our crawlers need time to completly collect all jobs from the jobboards in different timezones.</i>
        
    """
    url = f"https://daily-international-job-postings.p.rapidapi.com/api/v1/meta/jobs/countries"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-international-job-postings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_bulk_jobs(date: str, countrycode: str, source: str='monster_lu', earlybird: bool=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a paginated list of job postings for a query using countryCode, date, and optionally source. The results are paginated with max. 100 jobs.<br /><br /><i>Exemplary Call: `/api/v1/bulk/jobs?countryCode=lu&date=2022-02-01&source=linkedin_lu`<br /><br /><i>NOTE: only available for special subscription plans."
    date: A day starting in 01/2020 with the format 'YYYY-MM-DD'. <br /><br /><i>NOTE: the day must lie 2 days before 'now' as our crawlers need time to completly collect all jobs from the jobboards in different timezones.</i>
        countrycode: A 2-letter countryCode from ISO 3166-1 alpha-2. <br /><br /><i>NOTE: Internally, we lowerCase the countryCode and convert 'gb' into 'uk' and 'el' into 'gr'.</i>
        source: A source used by us constructed with in format '${portal}_${tld or countryCode}', e.g., 'monster_uk' or 'linkedin_lu'.
        earlybird: Indicates that job postings from the last two days are allowed<br /><br />WARN: the amount of jobs will change from time to time).<br /><br /><i>NOTE: only available for special subscription plans.
        page: The page number in the pagination starting at 0 (zero). (Defaults to 0)
        
    """
    url = f"https://daily-international-job-postings.p.rapidapi.com/api/v1/bulk/jobs"
    querystring = {'date': date, 'countryCode': countrycode, }
    if source:
        querystring['source'] = source
    if earlybird:
        querystring['earlybird'] = earlybird
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-international-job-postings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_bulk_jobs_ids(ids: str, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a paginated list of job postings for multiple ids, which were loaded, e.g., via a search. The results are paginated with max. 100 jobs.<br /><br /><i>Exemplary Call: `/api/v1/bulk/jobs/62029bba7d853d03ac262246,61f9b57fc328167ecf2c8a9c`<br /><br /><i>NOTE: only available for special subscription plans."
    ids: A comma-separated List of MongoDB ObjectId identifiers for job postings previously determined via a search.
        page: The page number in the pagination starting at 0 (zero). (Defaults to 0)
        
    """
    url = f"https://daily-international-job-postings.p.rapidapi.com/api/v1/bulk/jobs/{ids}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-international-job-postings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_jobs_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a job posting for an id, which was loaded via the search (e.g., by calling `/api/v1/jobs/62029bba7d853d03ac262246`)."
    id: A MongoDB ObjectId identifier for a job posting previously determined via a search.
        
    """
    url = f"https://daily-international-job-postings.p.rapidapi.com/api/v1/jobs/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-international-job-postings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_jobs(date: str, countrycode: str, includesource: bool=None, source: str='monster_lu', earlybird: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the search results for a query using countryCode, date, and optionally source.<br /><br /><i>Exemplary Call: `/api/v1/jobs?countryCode=lu&date=2022-02-01`"
    date: A day starting in 01/2020 with the format 'YYYY-MM-DD'. <br /><br /><i>NOTE: the day must lie 2 days before 'now' as our crawlers need time to completly collect all jobs from the jobboards in different timezones.</i>
        countrycode: A 2-letter countryCode from ISO 3166-1 alpha-2. <br /><br /><i>NOTE: Internally, we lowerCase the countryCode and convert 'gb' into 'uk' and 'el' into 'gr'.</i>
        includesource: Indicates that the source should be included for every job posting (WARN: result might get too big). <br /><br /><i>NOTE: only available for subscription plans PRO and above.
        source: A source used by us constructed with in format '${portal}_${tld or countryCode}', e.g., 'monster_uk' or 'linkedin_lu'.
        earlybird: Indicates that job postings from the last two days are allowed<br /><br />WARN: the amount of jobs will change from time to time.<br /><br /><i>NOTE: only available for special subscription plans.
        
    """
    url = f"https://daily-international-job-postings.p.rapidapi.com/api/v1/jobs"
    querystring = {'date': date, 'countryCode': countrycode, }
    if includesource:
        querystring['includeSource'] = includesource
    if source:
        querystring['source'] = source
    if earlybird:
        querystring['earlybird'] = earlybird
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daily-international-job-postings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

