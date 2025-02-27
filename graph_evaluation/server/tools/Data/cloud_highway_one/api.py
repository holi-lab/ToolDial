import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_inter_regional_latency(dstregion: str, srcregion: str, srcprovider: str, dstprovider: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### GET method to get the latency from a source region to a destination region
		
		Required parameters:
		- `srcProvider`: provider name, e.g. `aws`
		- `srcRegion`: region code, e.g. `us-west-2`
		- `dstProvider`: provider name, e.g. `aws`
		- `dstRegion`: region code, e.g. `ap-east-1`
		
		The latency has "directions", aka, switching source and destination region will get a different result (although they are super close)
		
		#### Example Query
		To get latency from `AWS us-west-2` region to `AWS ap-east-1`  region:
		
		` /getLatency?srcProvider=aws&srcRegion=us-west-2&dstProvider=aws&dstRegion=ap-east-1`
		
		#### Example Response (JSON format, latency in milliseconds):
		` { ping: 143.9680204 }`"
    
    """
    url = f"https://cloud-highway-one.p.rapidapi.com/getLatency"
    querystring = {'dstRegion': dstregion, 'srcRegion': srcregion, 'srcProvider': srcprovider, 'dstProvider': dstprovider, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cloud-highway-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_best_destination_region_from_a_source_region(srcprovider: str, srcregion: str, dstcandidate: str='aws@ap-east-1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### GET method to get the region with the lowest latency from a source region.
		
		Required parameters:
		- `srcProvider`: provider name, e.g. `aws`
		- `srcRegion`: region code, e.g. `us-west-2`
		
		Optional parameters:
		- `dstCandidate`: Use `@` to join destination provider name and region name, e.g. `aws@us-west-1`
		- `dstCandidate`: this parameter can be repeated for up to 100 times
		- `dstCandidate`: ...
		...
		
		
		You can specify up to **100** destination region candidates. If no candidates specified, it will check against **all** other supported regions from **all** providers.
		
		For each candidate, use `@` to join destination provider name and region name, e.g. `aws@us-west-1`.
		
		Put one destination candidate in each `dstCandidate` query key.
		
		#### Example Query
		 To check which of the three candidate destination regions `AWS us-west-1`, `AWS ap-east-1` and `AWS eu-central-1` has the lowest latency from source region `AWS us-west-2`:
		
		`/getBestDstRegion?srcProvider=aws&srcRegion=us-west-2&dstCandidate=aws@us-west-1&dstCandidate=aws@ap-east-1&dstCandidate=aws@eu-central-1`
		
		#### Example Response (JSON format, latency in milliseconds):
		`{
		   result: { dstProvider: 'aws', dstRegion: 'us-west-2', ping: 60.0498 }
		}
		`"
    dstcandidate: To add more candidates, just add another `dstCandidate` parameter.  The number of  `dstCandidate` parameters can be anywhere from 0 to 100.
        
    """
    url = f"https://cloud-highway-one.p.rapidapi.com/getBestDstRegion"
    querystring = {'srcProvider': srcprovider, 'srcRegion': srcregion, }
    if dstcandidate:
        querystring['dstCandidate'] = dstcandidate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cloud-highway-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_complete_latency_dataset_among_all_regions(acknowledgement: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### GET method to get the **entire** dataset (all possible permutations of latencies from each region to another including itself, in random order)
		
		Required parameter:
		- `acknowledgement`: use this **exact** string: `Yes_I_Understand_This_Operation_Is_Expensive_And_I_Should_Only_Make_The_Request_When_I_Really_Need_It`
		
		#### Example Query
		To get a complete latency dataset from each supported region to all regions (including itself):
		
		`/getAllData?acknowledgement=Yes_I_Understand_This_Operation_Is_Expensive_And_I_Should_Only_Make_The_Request_When_I_Really_Need_It`
		
		#### Example Response (**A LARGE JSON**, latency in milliseconds):
		```
		 {
		    data: [
		      { srcProvider: 'aws', srcRegion: 'us-west-2', dstProvider: 'aws', dstRegion: 'ap-east-1', ping: 125.74213 },
		      { srcProvider: 'aws', srcRegion: 'eu-central-1', dstProvider: 'aws', dstRegion: 'eu-central-1', ping: 20.00115 },
		      ...
		    ]
		 }
		```"
    acknowledgement: Must match exactly
        
    """
    url = f"https://cloud-highway-one.p.rapidapi.com/getAllData"
    querystring = {'acknowledgement': acknowledgement, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cloud-highway-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latencies_against_all_supported_regions_from_a_source_region(srcregion: str, srcprovider: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### GET method to get the latencies against all supported regions from a source region
		
		Required parameters:
		- `srcProvider`: provider name, e.g. `aws`
		- `srcRegion`: region code, e.g. `us-west-2`
		
		#### Example Query
		To get latencies from `AWS us-west-2` region to all supported regions:
		
		`/getAllDstRegion?srcProvider=aws&srcRegion=us-west-2`
		
		#### Example Response (JSON format, latency in milliseconds):
		```
		  {
		    data: [
		      { dstProvider: 'aws', dstRegion: 'ap-east-1', ping: 125.5481 },
		      { dstProvider: 'aws', dstRegion: 'eu-central-1', ping: 200.00018 },
		      ...
		    ]
		  }
		```"
    
    """
    url = f"https://cloud-highway-one.p.rapidapi.com/getAllDstRegion"
    querystring = {'srcRegion': srcregion, 'srcProvider': srcprovider, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cloud-highway-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

