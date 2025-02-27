import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def company_profile_endpoint(url: str, extra: str='include', categories: str='include', use_cache: str='if-present', acquisitions: str='include', funding_data: str='include', exit_data: str='include', resolve_numeric_id: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 1 credit / successful request.
		Get structured data of a Company Profile"
    url: 
    URL of the LinkedIn Company Profile to crawl.

    URL should be in the format of `https://www.linkedin.com/company/<public_identifier>`
    
        extra: 
    Enriches the Company Profile with extra details from external sources. Details include Crunchbase ranking, contact email, phone number, Facebook account, Twitter account, funding rounds and amount, IPO status, investor information, etc.
    
    Default value is `\\\"exclude\\\"`.
    The other acceptable value is `\\\"include\\\"`, which will include these extra details (if available) for `1` extra credit.
    
        categories: 
    Appends categories data of this company.

    Default value is `\\\"exclude\\\"`.
    The other acceptable value is `\\\"include\\\"`, which will include these categories (if available) for `1` extra credit.
    
        use_cache: `if-present` The default behavior. Fetches profile from cache regardless of age of profile. If profile is not available in cache, API will attempt to source profile externally.

`if-recent` API will make a best effort to return a fresh profile no older than 29 days.Costs an extra `1` credit on top of the cost of the base endpoint.
        acquisitions: 
    Provides further enriched data on acquisitions made by this company from external sources.

    Default value is `\\\"exclude\\\"`.
    The other acceptable value is `\\\"include\\\"`, which will include these acquisition data (if available) for `1` extra credit.
    
        funding_data: 
    Returns a list of funding rounds that this company has received.

    Default value is `\\\"exclude\\\"`.
    The other acceptable value is `\\\"include\\\"`, which will include these categories (if available) for `1` extra credit.
    
        exit_data: 
    Returns a list of investment portfolio exits.

    Default value is `\\\"exclude\\\"`.
    The other acceptable value is `\\\"include\\\"`, which will include these categories (if available) for `1` extra credit.
    
        resolve_numeric_id: 
    Enable support for Company Profile URLs with numerical IDs that you most frequently fetch from Sales Navigator. 
    We achieve this by resolving numerical IDs into vanity IDs with cached company profiles from [LinkDB](https://nubela.co/proxycurl/linkdb). 
    For example, we will turn `https://www.linkedin.com/company/1234567890` to `https://www.linkedin.com/company/acme-corp` -- for which the API endpoint only supports the latter.
    
    This parameter accepts the following values:
    - `false` (default value) - Will not resolve numerical IDs.
    - `true` - Enable support for Company Profile URLs with numerical IDs. 
    Costs an extra `2` credit on top of the base cost of the endpoint.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/company"
    querystring = {'url': url, }
    if extra:
        querystring['extra'] = extra
    if categories:
        querystring['categories'] = categories
    if use_cache:
        querystring['use_cache'] = use_cache
    if acquisitions:
        querystring['acquisitions'] = acquisitions
    if funding_data:
        querystring['funding_data'] = funding_data
    if exit_data:
        querystring['exit_data'] = exit_data
    if resolve_numeric_id:
        querystring['resolve_numeric_id'] = resolve_numeric_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_profile_picture_endpoint(linkedin_person_profile_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 0 credit / successful request.
		Get the profile picture of a person.
		
		Profile pictures are served from cached people profiles found within [LinkDB](https://nubela.co/proxycurl/linkdb).
		If the profile does not exist within [LinkDB](https://nubela.co/proxycurl/linkdb), then the API will return a `404` status code."
    linkedin_person_profile_url: 
    LinkedIn Profile URL of the person that you are trying to get the profile picture of.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/person/profile-picture"
    querystring = {'linkedin_person_profile_url': linkedin_person_profile_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_lookup_endpoint(company_domain: str, first_name: str, location: str='Seattle', last_name: str='Gates', title: str='Co-chair', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 2 credits / successful request.
		Resolve LinkedIn Profile"
    company_domain: Company name or domain
        first_name: First name of the user
        location: 
    The location of this user.

    Name of country, city or state.
    
        last_name: Last name of the user
        title: Title that user is holding at his/her current job
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/profile/resolve"
    querystring = {'company_domain': company_domain, 'first_name': first_name, }
    if location:
        querystring['location'] = location
    if last_name:
        querystring['last_name'] = last_name
    if title:
        querystring['title'] = title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def role_lookup_endpoint(company_name: str, role: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 3 credits / successful request.
		Finds the closest (person) profile with a given role in a Company.
		For example, you can use this endpoint to find the "CTO" of "Apple".
		This API endpoint returns only one result that is the closest match.
		
		There is also a [role search](https://nubela.co/blog/search-employees-with-employee-listing-api/)
		under the Employee Listing Endpoint if you require:
		
		* precision on the target company
		* a list of employees that matches a role (instead of one result)."
    company_name: Name of the company that you are searching for
        role: Role of the profile that you are lookin up
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/find/company/role"
    querystring = {'company_name': company_name, 'role': role, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reveal_endpoint(ip: str, role_contact_number: str='include', role_personal_email: str='include', role: str='ceo', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 2 credits / successful request.
		Deanonymize an IPv4 address and associate the Company behind the IPv4 address."
    ip: 
    The target IPv4 address.
    
        role_contact_number: 
    Append personal contact numbers to the response if the system finds a relevant person profile.
    
        role_personal_email: 
    Append personal email addresses to the response if the system finds a relevant person profile.
    
        role: 
    Lookup and append an employee of a certain role of the company.
    Within the same API call, You can choose to lookup a person with a given role within this organisation that you might want to reach out to.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/reveal/company"
    querystring = {'ip': ip, }
    if role_contact_number:
        querystring['role_contact_number'] = role_contact_number
    if role_personal_email:
        querystring['role_personal_email'] = role_personal_email
    if role:
        querystring['role'] = role
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_profile_endpoint(fallback_to_cache: str, url: str, skills: str='include', twitter_profile_id: str='include', extra: str='include', inferred_salary: str='include', facebook_profile_id: str='include', use_cache: str='if-present', personal_email: str='include', github_profile_id: str='include', personal_contact_number: str='include', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 1 credit / successful request.
		Get structured data of a Personal Profile"
    fallback_to_cache: 
    Tweaks the fallback behavior if an error arises from fetching a fresh profile.
    
    This parameter accepts the following values:
    * `on-error` (default value) - Fallback to reading the profile from cache if an error arises.
    * `never` - Do not ever read profile from cache.
    
        url: 
    URL of the LinkedIn Profile to crawl.

    URL should be in the format of `https://www.linkedin.com/in/<public-identifier>`
    
        skills: 
    Include skills data from external sources.

    This parameter accepts the following values:
    - `exclude` (default value) - Does not provide skills data field.
    - `include` - Append skills data to the person profile object. Costs an extra `1` credit on top of the cost of the base endpoint (if data is available).
    
        twitter_profile_id: 
    Enriches the Person Profile with Twitter Id from external sources.

    This parameter accepts the following values:
    - `exclude` (default value) - Does not provide Twitter Id data field.
    - `include` - Append Twitter Id data to the person profile object. Costs an extra `1` credit on top of the cost of the base endpoint (if data is available).
    
        extra: 
    Enriches the Person Profile with extra details from external sources. Extra details include gender, birth date, industry and interests.

    This parameter accepts the following values:
    - `exclude` (default value) - Does not provide extra data field.
    - `include` - Append extra data to the person profile object. Costs an extra `1` credit on top of the cost of the base endpoint (if data is available).
    
        inferred_salary: 
    Include inferred salary range from external sources.

    This parameter accepts the following values:
    - `exclude` (default value) - Does not provide inferred salary data field.
    - `include` - Append inferred salary range data to the person profile object. Costs an extra `1` credit on top of the cost of the base endpoint (if data is available).
    
        facebook_profile_id: 
    Enriches the Person Profile with Facebook Id from external sources.

    This parameter accepts the following values:
    - `exclude` (default value) - Does not provide Facebook Id data field.
    - `include` - Append Facebook Id data to the person profile object. Costs an extra `1` credit on top of the cost of the base endpoint (if data is available).
    
        use_cache: `if-present` The default behavior. Fetches profile from cache regardless of age of profile. If profile is not available in cache, API will attempt to source profile externally.

`if-recent` API will make a best effort to return a fresh profile no older than 29 days.Costs an extra `1` credit on top of the cost of the base endpoint.
        personal_email: 
    Enriches the Person Profile with personal emails from external sources.

    This parameter accepts the following values:
    - `exclude` (default value) - Does not provide personal emails data field.
    - `include` - Append personal emails data to the person profile object. Costs an extra `1` credit per email returned on top of the cost of the base endpoint (if data is available).
    
        github_profile_id: 
    Enriches the Person Profile with Github Id from external sources.

    This parameter accepts the following values:
    - `exclude` (default value) - Does not provide Github Id data field.
    - `include` - Append Github Id data to the person profile object. Costs an extra `1` credit on top of the cost of the base endpoint (if data is available).
    
        personal_contact_number: 
    Enriches the Person Profile with personal numbers from external sources.

    This parameter accepts the following values:
    - `exclude` (default value) - Does not provide personal numbers data field.
    - `include` - Append personal numbers data to the person profile object. Costs an extra `1` credit per email returned on top of the cost of the base endpoint (if data is available).
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/v2/linkedin"
    querystring = {'fallback_to_cache': fallback_to_cache, 'url': url, }
    if skills:
        querystring['skills'] = skills
    if twitter_profile_id:
        querystring['twitter_profile_id'] = twitter_profile_id
    if extra:
        querystring['extra'] = extra
    if inferred_salary:
        querystring['inferred_salary'] = inferred_salary
    if facebook_profile_id:
        querystring['facebook_profile_id'] = facebook_profile_id
    if use_cache:
        querystring['use_cache'] = use_cache
    if personal_email:
        querystring['personal_email'] = personal_email
    if github_profile_id:
        querystring['github_profile_id'] = github_profile_id
    if personal_contact_number:
        querystring['personal_contact_number'] = personal_contact_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def personal_contact_number_lookup_endpoint(linkedin_profile_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 1 credit / contact number returned.
		Given an LinkedIn profile, returns a list of personal contact numbers belonging to this identity."
    linkedin_profile_url: 
    LinkedIn Profile URL of the person you want to extract personal contact numbers from.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/contact-api/personal-contact"
    querystring = {'linkedin_profile_url': linkedin_profile_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse_work_email_lookup_endpoint(work_email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 3 credits / successful request.
		Resolve LinkedIn Profile from a work email address"
    work_email: Work email address of the user
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/profile/resolve/email"
    querystring = {'work_email': work_email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def work_email_lookup_endpoint(linkedin_profile_url: str, callback_url: str='https://webhook.site/29e12f17-d5a2-400a-9d08-42ee9d83600a', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 3 credits / request.
		Lookup work email address of a LinkedIn Person Profile.
		
		Email addresses returned are verified to not be role-based or catch-all emails. Email addresses
		returned by our API endpoint come with a 95+% deliverability guarantee
		
		**Endpoint behavior**
		
		*This endpoint* **_may not_** *return results immediately.*
		
		If you provided a webhook in your request parameter, our application will call your webhook with
		the result once. See `Webhook request` below."
    linkedin_profile_url: 
    Linkedin Profile URL of the person you want to
    extract work email address from.
    
        callback_url: 
    Webhook to notify your application when
    the request has finished processing.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/profile/email"
    querystring = {'linkedin_profile_url': linkedin_profile_url, }
    if callback_url:
        querystring['callback_url'] = callback_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_profile_picture_endpoint(linkedin_company_profile_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 0 credit / successful request.
		Get the profile picture of a company.
		
		Profile pictures are served from cached company profiles found within [LinkDB](https://nubela.co/proxycurl/linkdb).
		If the profile does not exist within [LinkDB](https://nubela.co/proxycurl/linkdb), then the API will return a `404` status code."
    linkedin_company_profile_url: 
    LinkedIn Profile URL of the company that you are trying to get the profile picture of.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/company/profile-picture"
    querystring = {'linkedin_company_profile_url': linkedin_company_profile_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def personal_email_lookup_endpoint(linkedin_profile_url: str, email_validation: str='include', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 1 credit / email returned.
		Given an LinkedIn profile, returns a list of personal emails belonging to this identity. Emails are verified to be deliverable."
    linkedin_profile_url: 
    LinkedIn Profile URL of the person you want to extract personal email addresses from.
    
        email_validation: 
    Perform deliverability validation on each email. (Costs 1 extra credit per email found).
    
    Takes the following values:
     * include - Perform email validation.
     * exclude (default) - Do not perform email validation.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/contact-api/personal-email"
    querystring = {'linkedin_profile_url': linkedin_profile_url, }
    if email_validation:
        querystring['email_validation'] = email_validation
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def employee_search_api_endpoint(linkedin_company_profile_url: str, keyword_regex: str, page_size: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 10 credits / successful request.
		Search employees of a target by their job title."
    linkedin_company_profile_url: 
    LinkedIn Profile URL of the target company.
    
        keyword_regex: 
    Job title keyword to search for in regular expression format.
    
        page_size: 
    Tune the maximum results returned per API call.
    The default value of this parameter is `200000`.
    Accepted values for this parameter is an integer ranging from `1` to `200000`.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/company/employee/search/"
    querystring = {'linkedin_company_profile_url': linkedin_company_profile_url, 'keyword_regex': keyword_regex, }
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def employee_count_endpoint(url: str, linkedin_employee_count: str='include', employment_status: str='current', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 1 credit / successful request.
		Get a number of total employees of a Company.
		
		Get an employee count of this company from various sources."
    url: 
    URL of the LinkedIn Company Profile to target.

    URL should be in the format of `https://www.linkedin.com/company/<public_identifier>`
    
        linkedin_employee_count: 
    Option to include a scraped employee count value from the target company's LinkedIn profile.

    Valid values are `include` and `exclude`:

    * `exclude` (default) : To exclude the scraped employee count.
    * `include` : To include the scraped employee count.

    Costs an extra `1` credit on top of the base cost of the endpoint.
    
        employment_status: 
    Parameter to tell the API to filter past or current employees.

    Valid values are `current`, `past`, and `all`:

    * `current` (default) : count current employees
    * `past` : count past employees
    * `all` : count current & past employees
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/company/employees/count/"
    querystring = {'url': url, }
    if linkedin_employee_count:
        querystring['linkedin_employee_count'] = linkedin_employee_count
    if employment_status:
        querystring['employment_status'] = employment_status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def disposable_email_address_check_endpoint(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 0 credit / request.
		Given an email address, checks if the email address belongs to a disposable email service."
    email: Email address to check
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/disposable-email"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jobs_listing_endpoint(search_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 2 credits / successful request.
		List jobs posted by a company on LinkedIn"
    search_id: 
    The `search_id` of the company on LinkedIn.
    You can get the `search_id` of a LinkedIn company via
    [Company Profile API](#company-api-company-profile-endpoint).
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/v2/linkedin/company/job"
    querystring = {'search_id': search_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def employee_listing_endpoint(url: str, enrich_profiles: str='enrich', page_size: str='100', employment_status: str='current', role_search: str='[Ff][Oo][Uu][Nn][Dd][Ee][Rr]', resolve_numeric_id: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 3 credits / employee returned.
		Get a list of employees of a Company.
		
		This API endpoint is limited by LinkDB which is populated with profiles in the US, UK, Canada, Israel, Australia, Ireland, New Zealand and Singapore.
		As such, this endpoint is best used to list employees working in companies based in those locations only."
    url: 
    URL of the LinkedIn Company Profile to target.

    URL should be in the format of `https://www.linkedin.com/company/<public_identifier>`
    
        enrich_profiles: 
    Get the full profile of employees instead of only their profile urls.

    Each request respond with a streaming response of profiles.

    The valid values are:
    
    * `skip` (default): lists employee's profile url
    * `enrich`: lists full profile of employees

    Calling this API endpoint with this parameter would add `1` credit per employee returned.
    
        page_size: 
    Tune the maximum results returned per API call.

    The default value of this parameter is `200000`.

    Accepted values for this parameter is an integer ranging from `1` to `200000`.

    When `enrich_profiles=enrich`, this parameter accepts value ranging from `1` to `100`.
    
        employment_status: 
    Parameter to tell the API to return past or current employees.

    Valid values are `current`, `past`, and `all`:

    * `current` (default) : lists current employees
    * `past` : lists past employees
    * `all` : lists current & past employees
    
        role_search: 
    Filter employees by their title by matching the employee's title against a *regular expression*.

    The default value of this parameter is `null`.

    The accepted value is a *regular expression* (regex).

    (The base cost of calling this API endpoint with this parameter would be `10` credits.
    Each employee matched and returned would cost `6` credits per employee returned.)
    
        resolve_numeric_id: 
    Enable support for Company Profile URLs with numerical IDs that you most frequently fetch from Sales Navigator. 
    We achieve this by resolving numerical IDs into vanity IDs with cached company profiles from [LinkDB](https://nubela.co/proxycurl/linkdb). 
    For example, we will turn `https://www.linkedin.com/company/1234567890` to `https://www.linkedin.com/company/acme-corp` -- for which the API endpoint only supports the latter.
    
    This parameter accepts the following values:
    - `false` (default value) - Will not resolve numerical IDs.
    - `true` - Enable support for Company Profile URLs with numerical IDs. 
    Costs an extra `2` credit on top of the base cost of the endpoint.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/company/employees/"
    querystring = {'url': url, }
    if enrich_profiles:
        querystring['enrich_profiles'] = enrich_profiles
    if page_size:
        querystring['page_size'] = page_size
    if employment_status:
        querystring['employment_status'] = employment_status
    if role_search:
        querystring['role_search'] = role_search
    if resolve_numeric_id:
        querystring['resolve_numeric_id'] = resolve_numeric_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_lookup_endpoint(company_domain: str='accenture.com', company_location: str='sg', company_name: str='Accenture', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 2 credits / successful request.
		Resolve Company LinkedIn Profile from company name,
		    domain name and location."
    company_domain: Company website or Company domain
Requires either `company_domain` or `company_name`
        company_location: 
    The location / region of company.
    ISO 3166-1 alpha-2 codes
    
        company_name: Company Name
Requires either `company_domain` or `company_name`
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/company/resolve"
    querystring = {}
    if company_domain:
        querystring['company_domain'] = company_domain
    if company_location:
        querystring['company_location'] = company_location
    if company_name:
        querystring['company_name'] = company_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_credit_balance_endpoint(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 0 credit / successful request.
		Get your current credit(s) balance"
    
    """
    url = f"https://proxycurl.p.rapidapi.com/api/credit-balance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def job_profile_endpoint(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 2 credits / successful request.
		Get structured data of a LinkedIn Job Profile"
    url: 
    URL of the LinkedIn Job Profile to target.

    URL should be in the format of
    `https://www.linkedin.com/jobs/view/<job_id>`.
    [Jobs Listing Endpoint](#jobs-api-jobs-listing-endpoint)
    can be used to retrieve a job URL.
    
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/job"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def school_profile_endpoint(url: str, use_cache: str='if-present', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cost: 1 credit / successful request.
		Get structured data of a LinkedIn School Profile"
    url: 
    URL of the LinkedIn School Profile to crawl.

    URL should be in the format of `https://www.linkedin.com/school/<public_identifier>`
    
        use_cache: `if-present` The default behavior. Fetches profile from cache regardless of age of profile. If profile is not available in cache, API will attempt to source profile externally.

`if-recent` API will make a best effort to return a fresh profile no older than 29 days.Costs an extra `1` credit on top of the cost of the base endpoint.
        
    """
    url = f"https://proxycurl.p.rapidapi.com/api/linkedin/school"
    querystring = {'url': url, }
    if use_cache:
        querystring['use_cache'] = use_cache
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proxycurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

