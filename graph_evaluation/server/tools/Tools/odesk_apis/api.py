import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_specific_task_record(code1_code2_coden_format: str, code1_code2_coden: str, company: int, team: int, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return details on a specific oTask or set of oTasks, this format can be used within a company (:companyid), team (:teamid) or user (:userid)"
    code1_code2_coden: The specific task codes, the list of codes should be separated with ";"
        format: formation of response
        company: The company ID
        team: The team ID
        username: The username of the target user account
        
    """
    url = f"https://odeskapis.p.rapidapi.com/https://www.odesk.com/api/otask/v1/tasks/companies/{company}/tasks/{code1_code2_coden}.{format} ,  https://www.odesk.com/api/otask/v1/tasks/companies/{company}/teams/{team}/tasks/{code1_code2_coden}.{format},  https://www.odesk.com/api/otask/v1/tasks/companies/{company}/teams/{team}/users/{username}/tasks/{code1_code2_coden}.{format}"
    querystring = {'code1-code2-coden-format': code1_code2_coden_format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_otask_records(company: int, format: str, team: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns all task records under the company"
    company: The company ID
        format: formation of response
        team: The team ID
        username: The username of the target user account
        
    """
    url = f"https://odeskapis.p.rapidapi.com/https://www.odesk.com/api/otask/v1/tasks/companies/{company}/tasks/full_list.{format},  https://www.odesk.com/api/otask/v1/tasks/companies/{company}/teams/{team}/tasks/full_list.{format},  https://www.odesk.com/api/otask/v1/tasks/companies/{company}/teams/{team}/users/{username}/tasks/full_list.{format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_my_jobs(buyer_team_reference: int, format: str, include_sub_teams: int=None, created_by: int=1234, status: str='open', created_time_from: str='2008-09-09 00:00:05', created_time_to: str='2009-01-20 11:59:55', page: str='20;10', order_by: str='created_time', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all jobs that a user has manage_recruiting access to. This API call can be used to find the reference ID of a specific job."
    buyer_team_reference: The buyer's team reference ID
        format: formation of response
        include_sub_teams: wether to include info about sub teams
        created_by: The user ID
        status: Status of Job
        created_time_from: Filter from time
        created_time_to: Filter to time
        page: Pagination, formed as $offset;$count
        order_by: Sorting
        
    """
    url = f"https://odeskapis.p.rapidapi.com/hr/v2/jobs.{format}"
    querystring = {'buyer_team__reference': buyer_team_reference, }
    if include_sub_teams:
        querystring['include_sub_teams'] = include_sub_teams
    if created_by:
        querystring['created_by'] = created_by
    if status:
        querystring['status'] = status
    if created_time_from:
        querystring['created_time_from'] = created_time_from
    if created_time_to:
        querystring['created_time_to'] = created_time_to
    if page:
        querystring['page'] = page
    if order_by:
        querystring['order_by'] = order_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generating_agency_specific_reports(format: str, tq: str, company: int, agency: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Time reports can be generated for an agency, in order to use this API the authorized user needs staffing manager permissions to the agency."
    format: note: format must be specified in tqx parameter, see example
        tq: The Google query goes here
        company: The company ID
        agency: The agency ID
        
    """
    url = f"https://odeskapis.p.rapidapi.com/https://www.odesk.com/gds/timereports/v1/companies/{company}/agencies/{agency}"
    querystring = {'format': format, 'tq': tq, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generating_company_wide_reports(format: str, tq: str, company: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Time reports can be generated on a company-wide level. All time reports fields are supported on this level except earnings related fields. In order to access this API the authorized user needs either hiring or finance permissions to all teams within the company."
    format: note: format must be specified in tqx parameter, see example
        tq: The Google query goes here
        company: The company ID
        
    """
    url = f"https://odeskapis.p.rapidapi.com/https://www.odesk.com/gds/timereports/v1/companies/{company}"
    querystring = {'format': format, 'tq': tq, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_job(job_reference_format: str, job_reference: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the complete job object for the referenced job, this is only available to users with manage_recruiting permissions within the team that the job is posted in."
    job_reference: Job reference ID
        format: formation of response
        
    """
    url = f"https://odeskapis.p.rapidapi.com/hr/v2/jobs/{job_reference}.{format}"
    querystring = {'job_reference-format': job_reference_format, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_tasks(company: int, format: str, team: int, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "pull all tasks assigned within a company, team or to a specific user"
    company: The company ID
        format: formation of response
        team: The team ID
        username: The username of the target user account.
        
    """
    url = f"https://odeskapis.p.rapidapi.com/https://www.odesk.com/api/otask/v1/tasks/companies/{company}/tasks.{format} https://www.odesk.com/api/otask/v1/tasks/companies/{company}/teams/{team}/tasks.{format},  https://www.odesk.com/api/otask/v1/tasks/companies/{company}/teams/{team}/users/{username}/tasks.{format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generating_provider_specific_reports(format: str, tq: str, provider: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API allows callers to fetch data source of themselves. No monetary fields, such as charges or earnings, are supported. The caller of this API must be the provider himself."
    format: note: format must be specified in tqx parameter, see example
        tq: The Google query goes here
        provider: The provider ID
        
    """
    url = f"https://odeskapis.p.rapidapi.com/https://www.odesk.com/gds/timereports/v1/providers/{provider}/hours,  https://www.odesk.com/gds/timereports/v1/providers/{provider}"
    querystring = {'format': format, 'tq': tq, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_time_reports_for_a_specific_team(format: str, tq: str, company: int, team: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Time reports can be generated for a specific team, with or without detailed monetary information based on the relationship of the authorized user at the time the call is made and what information is included in the call."
    format: note: format must be specified in tqx parameter, see example.
        tq: The Google query goes here.
        company: The company ID
        team: The team ID
        
    """
    url = f"https://odeskapis.p.rapidapi.com/https://www.odesk.com/gds/timereports/v1/companies/{company}/teams/{team}/hours,  https://www.odesk.com/gds/timereports/v1/companies/{company}/teams/{team}"
    querystring = {'format': format, 'tq': tq, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "odeskapis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

