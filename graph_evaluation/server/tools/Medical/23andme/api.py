import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def profile_picture(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    profile_id: profile_id as returned from /user/
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/profile_picture/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publish_get(profile_id: str, feature_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    profile_id: profile_id
        feature_id: feature_id
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/publish/{profile_id}/{feature_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def publish_get_by_link(profile_id: str, feature_id: str, link_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://23andme-23andme.p.rapidapi.com/publish/{profile_id}/{feature_id}/{link_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the user id, and a list of profiles (an account can have multiple genotyped people) with ids and whether or not they're genotyped. This endpoint is great for using an app anonymously because there is no personally identifying information."
    
    """
    url = f"https://23andme-23andme.p.rapidapi.com/user/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def traits(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our analysis for each profile for these traits (starred reports). trait is a value in possible_traits, or null if the profile's not analyzed at those markers."
    profile_id: profile_id
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/traits/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def neanderthal(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Estimated genome-wide proportion of Neanderthal ancestry for the user's profile. Most users have between 0.01 and 0.04 Neanderthal ancestry -- read a full explanation of the science. proportion is -1 for un-genotyped (or as-of-yet uncomputed) profiles."
    profile_id: profile_id
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/neanderthal/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def drug_responses(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our analysis of how each profile responds to these drugs (starred reports). status is reduced, typical, or increased for a person, not_applicable if the drug is not applicable to them (e.g., the oral contraceptives report is for women only), or null if they're not analyzed at any of the markers."
    profile_id: profile_id
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/drug_responses/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ancestry(threshold: str, profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ancestral breakdown for the user's profile. Each population has an absolute proportion of the genome. A population with sub_populations has an unassigned proportion — the ancestry we couldn't classify in it.   threshold is optional, default 0.75 and range (0.5, 1.0) exclusive. 0.51 means a speculative estimate, 0.75 standard, and 0.90 conservative. A higher threshold would increase the unassigned proportions, a lower would speculate."
    threshold: threshold ex:  0.9
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/ancestry/{profile_id}/"
    querystring = {'threshold': threshold, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genotypes(profile_id: str, locations: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the entire profile's genome as a packed string of base pairs "AACTGA...". This ~2MB request returns over a million SNP locations, so you must specify profile_id. If the profile has not yet unlocked certain SNPs, they'll be replaced with __. To know which SNP corresponds to which index in the string, see this key.  When our genotyping chip is upgraded, the packed string and corresponding key will grow, but the changes will be backwards-compatible additions."
    profile_id: profile_id
        locations: ex. rs3094315%20i3000001
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/genotypes/{profile_id}/"
    querystring = {}
    if locations:
        querystring['locations'] = locations
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def genomes(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the entire profile's genome as a packed string of base pairs "AACTGA...". This ~2MB request returns over a million SNP locations, so you must specify profile_id. If the profile has not yet unlocked certain SNPs, they'll be replaced with __. To know which SNP corresponds to which index in the string, see this key.  When our genotyping chip is upgraded, the packed string and corresponding key will grow, but the changes will be backwards-compatible additions."
    profile_id: profile_id
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/genomes/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def risks(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our analysis for each profile's lifetime risk of these diseases (starred reports). population_risk is the average risk for the population for which the analysis applies, and risk is the profile's risk."
    profile_id: profile_id
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/risks/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relatives(profile_id: str, limit: str=None, offset: str=None, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Relatives on 23andMe, for the user's profile. shared_segments is the total number of shared IBD segments; similarity is the actual proportion (1.00 being you, or an identical twin). maternal/paternal_side are True if this match is a relative from your mom or dad's side. range is defined if we couldn't pinpoint the relatedness of the match. match_id is a unique identifier.  Since you could have thousands of matches, limit defaults to 10, and offset to 0. count gives the total number of matches. Results are sorted by updated, descending. You can also get results that have been updated or added since a timestamp."
    profile_id: profile_id
        limit: limit
        offset: offset
        since: since
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/relatives/{profile_id}/"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def haplogroups(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For the user's profile, gets maternal and paternal haplogroups, as well as terminal SNPs. Maternal terminal SNPs include the rsid and rCRS reference position, while the paternal ones include the rsid and ISOGG SNP.  Note: if the profile belongs to a female, the paternal (y) haplogroup and terminal SNP information will be null."
    profile_id: profile_id
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/haplogroups/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carriers(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our analysis of whether or not each profile is a carrier for these diseases (starred reports). The person has 0, 1, or 2 mutations, or null if they're not analyzed at any of the markers. Normally, with one mutation, the person is considered a "carrier" and can pass the mutation to children; with two, the person is likely to be affected by the disease."
    
    """
    url = f"https://23andme-23andme.p.rapidapi.com/carriers/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def names(profile_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For the user and user's profile, gets first and last names. If your user wants to remain anonymous, you shouldn't request this scope. You can optionally filter by profile_id to get the names for just that profile."
    profile_id: profile_id as returned from /user/
        
    """
    url = f"https://23andme-23andme.p.rapidapi.com/names/{profile_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "23andme-23andme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

