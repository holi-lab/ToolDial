import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_route(payload: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a list of locations, the route service provides details about the trip,
		including locations, a summary with basic information about the entire trip, and a list of legs.
		
		**Locations**
		
		You specify locations as an ordered list of two or more locations within a JSON array. Locations are visited in the order specified.
		
		A location must include a latitude and longitude in decimal degrees. The coordinates can come from many input sources, such as a GPS location, a point or a click on a map, a geocoding service, and so on.
		
		To build a route, you need to specify two `break` locations. In addition, you can include `through`, `via` or `break_through` locations to influence the route path.
		
		<div class="tg-wrap">
		<table>
		<thead>
		<tr>
		    <th style="min-width: 128px;">Location parameters</th>
		    <th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		    <td><code>lat</code></td>
		    <td>
		Latitude of the location in degrees. This is assumed to be both the routing location and the display location if no <code>display_lat</code> and <code>display_lon</code> are provided.
		    </td>
		</tr>
		<tr>
		    <td><code>lon</code></td>
		    <td>
		Longitude of the location in degrees. This is assumed to be both the routing location and the display location if no <code>display_lat</code> and <code>display_lon</code> are provided.
		    </td>
		</tr>
		<tr>
		    <td><code>type</code></td>
		    <td>
			Type of location, either <code>break</code>, <code>through</code>, <code>via</code> or <code>break_through</code>. Each type controls two characteristics: whether or not to allow a u-turn at the location and whether or not to generate guidance/legs at the location. A <code>break</code> is a location at which we allows u-turns and generate legs and arrival/departure maneuvers. A <code>through</code> location is a location at which we neither allow u-turns nor generate legs or arrival/departure maneuvers. A <code>via</code> location is a location at which we allow u-turns but do not generate legs or arrival/departure maneuvers. A <code>break_through</code> location is a location at which we do not allow u-turns but do generate legs and arrival/departure maneuvers. If no type is provided, the type is assumed to be a <code>break</code>. The types of the first and last locations are ignored and are treated as breaks.
		    </td>
		</tr>
		</tbody>
		</table>
		</div>
		
		**Costing models**
		
		The routing service uses dynamic, run-time costing to generate the route path. The route request must include the name of the costing model and can include optional parameters available for the chosen costing model.
		
		<div class="tg-wrap">
		<table>
		<thead>
		<tr>
		    <th style="min-width: 128px;">Costing model</th>
		    <th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		    <td><code>auto</code></td>
		    <td>
			Standard costing for driving routes by car, motorcycle, truck, and so on that obeys automobile driving rules, such as access and turn restrictions. <code>Auto</code> provides a short time path (though not guaranteed to be shortest time) and uses intersection costing to minimize turns and maneuvers or road name changes. Routes also tend to favor highways and higher classification roads, such as motorways and trunks.
		    </td>
		</tr>
		<tr>
		    <td><code>bicycle</code></td>
		    <td>
			Standard costing for travel by bicycle, with a slight preference for using <a href="http://wiki.openstreetmap.org/wiki/Key:cycleway" target="_blank">cycleways</a> or roads with bicycle lanes. Bicycle routes follow regular roads when needed, but avoid roads without bicycle access.
		    </td>
		</tr>
		<tr>
		    <td><code>bus</code></td>
		    <td>
			Standard costing for bus routes. Bus costing inherits the auto costing behaviors, but checks for bus access on the roads.
		    </td>
		</tr>
		</tbody>
		</table>
		</div>
		"
    payload: Location information
        
    """
    url = f"https://spatialscale-route-engine.p.rapidapi.com/v1/route_engine/route"
    querystring = {'payload': payload, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spatialscale-route-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_isochrone(payload: str, is_id: str='Walk_From_Office', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The isochrone service computes areas that are reachable within specified time intervals from a location,
		and returns the reachable regions as contours of polygons or lines that you can display on a map.
		
		**Costing parameters**
		
		The isochrone service uses the auto, bicycle, pedestrian, and multimodal costing models available in the Turn-by-Turn service.
		Refer to the route costing models and costing options documentation for more on how to specify this input.
		
		**Other request parameters**
		
		<div class="tg-wrap">
		<table>
		<thead>
		<tr>
		    <th style="min-width: 128px;">Parameter</th>
		    <th>Description</th>
		</tr>
		</thead>
		<tbody>
		<tr>
		    <td><code>date_time</code></td>
		    <td>
		        The local date and time at the location. These parameters apply only for multimodal requests and are not used with other costing methods.
		        <ul>
		            <li><code>type</code></li>
		            <ul>
		                <li>0 - Current departure time for multimodal requests.</li>
		                <li>1 - Specified departure time for multimodal requests.</li>
		                <li>2 - Specified arrival time. Note: This is not yet implemented.</li>
		            </ul>
		            <li><code>value</code> - the date and time specified in ISO 8601 format (YYYY-MM-DDThh:mm) in the local time zone of departure or arrival. For example, "2016-07-03T08:06".
		            </li>
		        </ul>
		    </td>
		</tr>
		<tr>
		    <td><code>id</code></td>
		    <td>
		        Name of the isochrone request. If <code>id</code> is specified, the name is returned with the response.
		    </td>
		</tr>
		<tr>
		    <td><code>contours</code></td>
		    <td>
		        A JSON array of contour objects with the time in minutes or distance in kilometers and color to use for each isochrone contour. You can specify up to four contours (by default).
		        <ul>
		            <li><code>time</code> - A floating point value specifying the time in minutes for the contour.
		            </li>
		            <li><code>distance</code> - A floating point value specifying the distance in kilometers for the contour.
		            </li>
		            <li><code>color</code> -The color for the output of the contour. Specify it as a Hex value, but without the <code>#</code>, such as <code>"color":"ff0000"</code> for red. If no color is specified, the isochrone service will assign a default color to the output.
		            </li>
		        </ul>
		        You can only specify one metric per contour, i.e. time or distance.
		    </td>
		</tr>
		<tr>
		    <td><code>polygons</code></td>
		    <td>
		        A Boolean value to determine whether to return geojson polygons or linestrings as the contours. The default is <code>false</code>, which returns lines; when <code>true</code>, polygons are returned. Note: When <code>polygons</code> is <code>true</code>, any contour that forms a ring is returned as a polygon.
		    </td>
		</tr>
		<tr>
		    <td><code>generalize</code></td>
		    <td>
		        A floating point value in meters used as the tolerance for Douglas-Peucker generalization. Note: Generalization of contours can lead to self-intersections, as well as intersections of adjacent contours.
		    </td>
		</tr>
		</tbody>
		</table>
		</div>
		"
    payload: Location information
        is_id: Result identifier
        
    """
    url = f"https://spatialscale-route-engine.p.rapidapi.com/v1/route_engine/isochrone"
    querystring = {'payload': payload, }
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spatialscale-route-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

