import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_details_for_job(companyid: str, openingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    companyid: Company ID that posted job
        openingid: Job ID
        
    """
    url = f"https://harryanderson35-odesktemp.p.rapidapi.com/client/companies/{companyid}/openings/{openingid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "harryanderson35-odesktemp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_job_for_a_team(buyerteamreference: int, companyid: str, team: str, includesubteams: str='0|1', createdby: str='1234', status: str='open', createdtimefrom: str='2008-09-09T00:00:01', createdtimeto: str='2009-01-20T11:59:59', page: str='20;10', orderby: str='created_time', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    buyerteamreference: The buyer's team reference ID
        companyid: Company ID
        team: Team ID
        includesubteams: wether to include info about sub teams
        createdby: The user ID
        status: Status of Job
        createdtimefrom: Filter from time
        createdtimeto: Filter to time
        page: Pagination, formed as $offset;$count
        orderby: Sorting
        
    """
    url = f"https://harryanderson35-odesktemp.p.rapidapi.com/client/companies/{companyid}/team/{team}/openings/"
    querystring = {'buyerTeamReference': buyerteamreference, }
    if includesubteams:
        querystring['includeSubTeams'] = includesubteams
    if createdby:
        querystring['createdBy'] = createdby
    if status:
        querystring['status'] = status
    if createdtimefrom:
        querystring['createdTimeFrom'] = createdtimefrom
    if createdtimeto:
        querystring['createdTimeTo'] = createdtimeto
    if page:
        querystring['page'] = page
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "harryanderson35-odesktemp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_job_for_a_company(companyid: str, createdby: str='1234', status: str='open', createdtimefrom: str='2008-09-09T00:00:01', createdtimeto: str='2009-01-20T11:59:59', page: str='20;10', orderby: str='created_time', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    createdby: The user ID
        status: Status of Job
        createdtimefrom: Filter from time
        createdtimeto: Filter to time
        page: Pagination, formed as $offset;$count
        orderby: Sorting
        
    """
    url = f"https://harryanderson35-odesktemp.p.rapidapi.com/client/tractor/companies/{companyid}/openings/"
    querystring = {}
    if createdby:
        querystring['createdBy'] = createdby
    if status:
        querystring['status'] = status
    if createdtimefrom:
        querystring['createdTimeFrom'] = createdtimefrom
    if createdtimeto:
        querystring['createdTimeTo'] = createdtimeto
    if page:
        querystring['page'] = page
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "harryanderson35-odesktemp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get job categories"
    
    """
    url = f"https://harryanderson35-odesktemp.p.rapidapi.com/public/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "harryanderson35-odesktemp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_specific_jobs(openingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "See the details of a specific job opening"
    openingid: job opening ID
        
    """
    url = f"https://harryanderson35-odesktemp.p.rapidapi.com/public/openings/{openingid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "harryanderson35-odesktemp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_jobs(q: str='php', c1: str='Web Developing', c2: str='Web Design', qs: str='2d-design', to: str='magento', fb: int=4, min: int=1000, max: int=5000, t: str='"Hourly", "Fixed"', wl: int=20, dur: int=13, dp: str='07-22-2009', st: str='Open', tba: int=5, gr: str='PGroup', page: str='0;20', sort: str='date_posted;A', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for job openings"
    q: Search query
        c1: Category name, for now the full list of categories
        c2: Subcategory, which is related to category (c1)
        qs: Skill
        to: Search in titles only
        fb: Limit your search to buyers with at least a score of the number passed in this parameter.
        min: Minimal budget
        max: Maximal budget
        t: Job type
        wl: Hours per week; As Needed < 10 Hours/Week = '0',  Part Time: 10-30 hrs/week = '20',  Full Time: 30+ hrs/week = '40'
        dur: Engagement duration; Ongoing / More than 6 months = '26' 3 to 6 months = '13' 1 to 3 months = '4' Less than 1 month = '1' Less than 1 week = '0'
        dp: Date posted
        st: Status for search; Open Jobs = 'Open' Jobs in Progress = 'In Progress' Completed Jobs = 'Completed' Canceled Jobs = 'Cancelled'
        tba: Total billed assignments
        gr: Prefered group
        page: Paging, based on offset;count, e.g. 0;10 - return 10 records, starting from 0
        sort: Sorting, in format $field_name1;$field_name2;..$field_nameN;AD...A, where A eq. asc, D eq. desc, e.g. date_posted;A
        
    """
    url = f"https://harryanderson35-odesktemp.p.rapidapi.com/public/openings/"
    querystring = {}
    if q:
        querystring['q'] = q
    if c1:
        querystring['c1'] = c1
    if c2:
        querystring['c2'] = c2
    if qs:
        querystring['qs'] = qs
    if to:
        querystring['to'] = to
    if fb:
        querystring['fb'] = fb
    if min:
        querystring['min'] = min
    if max:
        querystring['max'] = max
    if t:
        querystring['t'] = t
    if wl:
        querystring['wl'] = wl
    if dur:
        querystring['dur'] = dur
    if dp:
        querystring['dp'] = dp
    if st:
        querystring['st'] = st
    if tba:
        querystring['tba'] = tba
    if gr:
        querystring['gr'] = gr
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "harryanderson35-odesktemp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of regions"
    
    """
    url = f"https://harryanderson35-odesktemp.p.rapidapi.com/public/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "harryanderson35-odesktemp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

