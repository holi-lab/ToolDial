import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trip_scurrentstatus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>You can use this endpoint to fetch the details of a trip which also includes the last known location.</p>
		<p><strong>Body Parameters</strong></p>
		
		<p><strong>Notes</strong></p>
		<ul>
		<li>Atleast one of the below three parameters should be passed</li>
		</ul>
		
		<div class="click-to-expand-wrapper is-table-wrapper"><table>
		<thead>
		<tr>
		<th>Parameter</th>
		<th>Mandatory</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		
		
		<tr>
		<td>invoice</td>
		<td>No</td>
		<td> invoice received in response of submit trip API</td>
		</tr>
		
		<tr>
		<td>id</td>
		<td>No</td>
		<td> trip_id received in response of submit trip API</td>
		</tr>
		
		<tr>
		<td>lr_number</td>
		<td>No</td>
		<td> lr_number received in response of submit trip API</td>
		</tr>
		
		</tbody>
		</table>
		
		
		<h2 id="output">OUTPUT</h2>
		<p>
		<h4>Invalid user credentials</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">{
		    "errors": "Invalid username/password."
		}
		</code></pre>"
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v3/trip/info/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchalltrips(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is for fetching all trips "
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v3/trips/fetch/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkconsent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>You can use this endpoint to check concent status of the passed mobile number.</p>
		<p><strong>Body Parameters</strong></p>
		
		<div class="click-to-expand-wrapper is-table-wrapper"><table>
		<thead>
		<tr>
		<th>Parameter</th>
		<th>Mandatory</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		
		
		<tr>
		<td>tel</td>
		<td>Yes</td>
		<td> 10 digit comma seperated mobile number used for creating trip</td>
		</tr>
		
		
		</tbody>
		</table>
		
		
		<h2 id="output">OUTPUT</h2>
		<p>
		<h4>Invalid user credentials</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">{
		    "errors": "Invalid username/password."
		}
		</code></pre>"
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v3/check_consent/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchatrip(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>This endpoint is for fetching a trip</p>
		<p><strong>Body Parameters</strong></p>
		
		<p><strong>Notes</strong></p>
		<ul>
		<li>Atleast one of the below three parameters should be passed</li>
		</ul>
		
		<div class="click-to-expand-wrapper is-table-wrapper"><table>
		<thead>
		<tr>
		<th>Parameter</th>
		<th>Mandatory</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		
		
		<tr>
		<td>invoice</td>
		<td>No</td>
		<td> invoice received in response of submit trip API</td>
		</tr>
		
		<tr>
		<td>trip_id</td>
		<td>No</td>
		<td> trip_id received in response of submit trip API</td>
		</tr>
		
		<tr>
		<td>lr_number</td>
		<td>No</td>
		<td> lr_number received in response of submit trip API</td>
		</tr>
		
		</tbody>
		</table>
		
		
		<h2 id="output">OUTPUT</h2>
		<p>
		<h4>Invalid user credentials</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">{
		    "errors": "Invalid username/password."
		}
		</code></pre>"
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v3/trip/fetch/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetchlocation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>This endpoint is for fetching location of a trip</p>
		<p><strong>Required Body Parameters</strong></p>
		
		<p><strong>Notes</strong></p>
		<ul>
		<li>Atleast one of the below three parameters should be passed</li>
		</ul>
		
		<div class="click-to-expand-wrapper is-table-wrapper"><table>
		<thead>
		<tr>
		<th>Parameter</th>
		<th>Mandatory</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		
		
		<tr>
		<td>invoice</td>
		<td>No</td>
		<td> invoice received in response of submit trip API</td>
		</tr>
		
		<tr>
		<td>id</td>
		<td>No</td>
		<td> trip_id received in response of submit trip API</td>
		</tr>
		
		<tr>
		<td>lr_number</td>
		<td>No</td>
		<td> lr_number received in response of submit trip API</td>
		</tr>
		
		</tbody>
		</table>
		
		
		<h2 id="output">OUTPUT</h2>
		<p>
		<h4>Invalid user credentials</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">{
		    "errors": "Invalid username/password."
		}
		</code></pre>"
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v3/trip/location/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def numberlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "
		<p>All  of <b>last updated locations</b> of provided concents phone numbers can be fetched through this endpoint,</p>
		
		<h2 id="output">OUTPUT</h2>
		<p>
		<h4>Invalid user credentials</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">{
		    "errors": "Invalid username/password."
		}
		</code></pre>
		"
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v2/number_list/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlatestlocation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>This API can be used for getting latest location of the passed number.</p>
		
		
		<p><strong>Body Parameters</strong></p>
		
		<div class="click-to-expand-wrapper is-table-wrapper"><table>
		<thead>
		<tr>
		<th>Parameter</th>
		<th>Mandatory</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		
		
		<tr>
		<td>phone_number</td>
		<td>Yes</td>
		<td> 10 digit previously added mobile number</td>
		</tr>
		
		
		</tbody>
		</table>
		
		
		<h2 id="output">OUTPUT</h2>
		<p>
		<h4>Invalid user credentials</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">{
		    "errors": "Invalid username/password."
		}
		</code></pre>
		
		<p>
		<h4>Wrong or not added mobile number passed</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">
		{
		    "status": "invalid_mobile"
		}
		</code></pre>"
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v2/location/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkconcent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>This endpoint is responsible for refreshing the tracking concent of provided number, i.e. whether location tracking is allowed or not yet</p>
		
		<p><strong>Body Parameters</strong></p>
		
		<div class="click-to-expand-wrapper is-table-wrapper"><table>
		<thead>
		<tr>
		<th>Parameter</th>
		<th>Mandatory</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		
		
		<tr>
		<td>tel</td>
		<td>Yes</td>
		<td> 10 digit previously added mobile number</td>
		</tr>
		
		
		</tbody>
		</table>
		
		
		<h2 id="output">OUTPUT</h2>
		<p>
		<h4>Invalid user credentials</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">{
		    "errors": "Invalid username/password."
		}
		</code></pre>
		
		<p>
		<h4>Wrong or not added mobile number passed</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">
		{
		    "status": "invalid_mobile"
		}
		</code></pre>"
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v2/check_consent/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def routehistory(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<p>Through this endpoint <b>route history</b> can be fetched of provided number..</p>
		
		<p><strong>Body Parameters</strong></p>
		
		<div class="click-to-expand-wrapper is-table-wrapper"><table>
		<thead>
		<tr>
		<th>Parameter</th>
		<th>Mandatory</th>
		<th>Description</th>
		</tr>
		</thead>
		<tbody>
		
		
		<tr>
		<td>phone_number</td>
		<td>Yes</td>
		<td> 10 digit previously added mobile number</td>
		</tr>
		
		<tr>
		<td>from_date</td>
		<td>Yes</td>
		<td>From date for Route history in YYYY/MM/DD formate</td>
		</tr>
		
		<tr>
		<td>to_date</td>
		<td>Yes</td>
		<td>To date for Route history in YYYY/MM/DD formate</td>
		</tr>
		
		</tbody>
		</table>
		
		
		<h2 id="output">OUTPUT</h2>
		<p>
		<h4>Invalid user credentials</h4>
		</p>
		<pre class="click-to-expand-wrapper is-snippet-wrapper language-undefined"><code class="is-highlighted language-undefined">{
		    "errors": "Invalid username/password."
		}
		</code></pre>"
    
    """
    url = f"https://sim-based-location-tracking1.p.rapidapi.com/api/v2/route_history/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sim-based-location-tracking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

