import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bounds_based_image(width: int, mapid: str, minx: int, stroke: str=None, fill: str=None, encodedpath: str=None, padding: int=None, attribution: str='bottomright', path: str=None, markers: str=None, latlng: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a raster image based on the given bounds."
    width: Width of the image in pixels.
        mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        minx: Longitude of the left (west) edge.
        stroke: Color to use as a stroke when drawing polygons. Deprecated, use \"path\" instead.
        fill: Color to use as a fill when drawing polygons. Deprecated, use \"path\" instead.
        encodedpath: Path in [Google Encoded Polyline Format](https://developers.google.com/maps/documentation/utilities/polylinealgorithm). Deprecated, use \"path\" instead.
        padding: Ensures the autofitted bounds or features are comfortably visible in the resulting area. E.g. use 0.1 to add 10% margin (at least) of the size to each side.
        width: Width of the stroke line when drawing polygons (in pixels). Deprecated, use \"path\" instead.
        attribution: Changes the position of map attribution. If you disable the attribution make sure to display it in your application yourself (visibly).
        path: Define path(s) to be drawn on top of the map. Can be used multiple times. See [https://support.maptiler.com/i26-static-maps-for-your-web](https://support.maptiler.com/i26-static-maps-for-your-web).
        markers: Define marker(s) to be drawn on top of the map. Can be used multiple times. See [https://support.maptiler.com/i26-static-maps-for-your-web](https://support.maptiler.com/i26-static-maps-for-your-web).
        latlng: Use [latitude, longitude] order for coordinates instead of [longitude, latitude].
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/static/{minx},{miny},{maxx},{maxy}/{width}x{height}{scale}.{format}"
    querystring = {}
    if stroke:
        querystring['stroke'] = stroke
    if fill:
        querystring['fill'] = fill
    if encodedpath:
        querystring['encodedpath'] = encodedpath
    if padding:
        querystring['padding'] = padding
    if width:
        querystring['width'] = width
    if attribution:
        querystring['attribution'] = attribution
    if path:
        querystring['path'] = path
    if markers:
        querystring['markers'] = markers
    if latlng:
        querystring['latlng'] = latlng
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def raster_xyz_tiles(y: int, mapid: str, z: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rasterized tiles (XYZ) of the map. Can be used with various libraries to display a raster map (e.g. Leaflet, OpenLayers, ...). It's usually better (if possible) to use the TileJSON rather than using the tile URL directly."
    mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/{tileSize/}/{z}/{x}/{y}{scale}.{format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def map_symbols_sprites(mapid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Map symbols (sprites) required to display the vector map."
    mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/sprite{scale}.{format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tilejson(mapid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TileJSON describing the metadata of the map as well as link to the XYZ tiles. Can be used with various libraries to display a raster map (e.g. Leaflet, OpenLayers, ...)."
    mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/{tileSize/}/tiles.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transform_coordinates(coordinates: str, t_srs: int=4326, ops: str='1623', s_srs: int=4326, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Transform coordinates"
    coordinates: List of coordinate pairs seperated by ; delimeter (Max 50 pairs).
        t_srs: Target CRS
        ops: List of codes of operations seperated by a | (pipe) operator
        s_srs: Source CRS
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/coordinates/transform/{coordinates}.json"
    querystring = {}
    if t_srs:
        querystring['t_srs'] = t_srs
    if ops:
        querystring['ops'] = ops
    if s_srs:
        querystring['s_srs'] = s_srs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_fitted_image(mapid: str, width: int, attribution: str='bottomright', latlng: bool=None, padding: int=None, fill: str=None, encodedpath: str=None, markers: str=None, stroke: str=None, path: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a raster image based on the given features. The area is calculated so that all the paths and markers given in query are visible."
    mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        width: Width of the image in pixels.
        attribution: Changes the position of map attribution. If you disable the attribution make sure to display it in your application yourself (visibly).
        width: Width of the stroke line when drawing polygons (in pixels). Deprecated, use \"path\" instead.
        latlng: Use [latitude, longitude] order for coordinates instead of [longitude, latitude].
        padding: Ensures the autofitted bounds or features are comfortably visible in the resulting area. E.g. use 0.1 to add 10% margin (at least) of the size to each side.
        fill: Color to use as a fill when drawing polygons. Deprecated, use \"path\" instead.
        encodedpath: Path in [Google Encoded Polyline Format](https://developers.google.com/maps/documentation/utilities/polylinealgorithm). Deprecated, use \"path\" instead.
        markers: Define marker(s) to be drawn on top of the map. Can be used multiple times. See [https://support.maptiler.com/i26-static-maps-for-your-web](https://support.maptiler.com/i26-static-maps-for-your-web).
        stroke: Color to use as a stroke when drawing polygons. Deprecated, use \"path\" instead.
        path: Define path(s) to be drawn on top of the map. Can be used multiple times. See [https://support.maptiler.com/i26-static-maps-for-your-web](https://support.maptiler.com/i26-static-maps-for-your-web).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/static/auto/{width}x{height}{scale}.{format}"
    querystring = {}
    if attribution:
        querystring['attribution'] = attribution
    if width:
        querystring['width'] = width
    if latlng:
        querystring['latlng'] = latlng
    if padding:
        querystring['padding'] = padding
    if fill:
        querystring['fill'] = fill
    if encodedpath:
        querystring['encodedpath'] = encodedpath
    if markers:
        querystring['markers'] = markers
    if stroke:
        querystring['stroke'] = stroke
    if path:
        querystring['path'] = path
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def center_based_image(lon: int, mapid: str, width: int, stroke: str=None, markers: str=None, attribution: str='bottomright', path: str=None, encodedpath: str=None, fill: str=None, latlng: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a raster image based on the specified center and zoom level."
    lon: Longitude of the center of the image.
        mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        width: Width of the image in pixels.
        stroke: Color to use as a stroke when drawing polygons. Deprecated, use \"path\" instead.
        width: Width of the stroke line when drawing polygons (in pixels). Deprecated, use \"path\" instead.
        markers: Define marker(s) to be drawn on top of the map. Can be used multiple times. See [https://support.maptiler.com/i26-static-maps-for-your-web](https://support.maptiler.com/i26-static-maps-for-your-web).
        attribution: Changes the position of map attribution. If you disable the attribution make sure to display it in your application yourself (visibly).
        path: Define path(s) to be drawn on top of the map. Can be used multiple times. See [https://support.maptiler.com/i26-static-maps-for-your-web](https://support.maptiler.com/i26-static-maps-for-your-web).
        encodedpath: Path in [Google Encoded Polyline Format](https://developers.google.com/maps/documentation/utilities/polylinealgorithm). Deprecated, use \"path\" instead.
        fill: Color to use as a fill when drawing polygons. Deprecated, use \"path\" instead.
        latlng: Use [latitude, longitude] order for coordinates instead of [longitude, latitude].
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/static/{lon},{lat},{zoom}/{width}x{height}{scale}.{format}"
    querystring = {}
    if stroke:
        querystring['stroke'] = stroke
    if width:
        querystring['width'] = width
    if markers:
        querystring['markers'] = markers
    if attribution:
        querystring['attribution'] = attribution
    if path:
        querystring['path'] = path
    if encodedpath:
        querystring['encodedpath'] = encodedpath
    if fill:
        querystring['fill'] = fill
    if latlng:
        querystring['latlng'] = latlng
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tiles_tilejson(tilesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TileJSON describing the metadata of the tiles as well as link to the XYZ tiles. Can be used with various libraries to display the tiles (e.g. Leaflet, OpenLayers, ...)."
    tilesid: Identifier of the tiles. See [MapTiler Cloud Tiles](https://cloud.maptiler.com/tiles/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/tiles/{tilesid}/tiles.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tiles_embeddable_html_viewer(tilesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Embeddable HTML viewer"
    tilesid: Identifier of the tiles. See [MapTiler Cloud Tiles](https://cloud.maptiler.com/tiles/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/tiles/{tilesid}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_coordinate_systems(query: str, transformations: bool=None, exports: bool=None, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search coordinate systems"
    query: Query string used to search the catalog. It accepts plain terms such as names of locations and key:value pairs for filtering.

Following parameter values can be used:
- kind
    - \\\\* (All kinds)
    - CRS (All coordinate reference systems - default)
        - CRS-PROJCRS (Projected coordinate systems)
        - CRS-GEOGCRS (Geodetic coordinate systems)
        - CRS-GEOG3DCRS (Geodetic 3D coordinate systems)
        - CRS-GCENCRS (Geocentric coordinate systems)
        - CRS-VERTCRS (Vertical coordinate systems)
        - CRS-ENGCRS (Engineering coordinate systems)
        - CRS-COMPOUNDCRS (Compound coordinate systems)
        - CRS-DRVDCRS (Derived coordinate systems)
    - COORDOP (All operations)
        - COORDOP-COPTRAN (Transformations)
        - COORDOP-COPCONO (Compound operations)
        - COORDOP-POIMOTO (Point motion operations)
        - COORDOP-COPCON (Conversions)
    - DATUM (All datums)
        - DATUM-VERTDAT (Vertical datums)
        - DATUM-ENGDAT (Engineering datums)
        - DATUM-GEODDAT (Geodetic datums)
        - DATUM-DYNGEODDA (Dynamic geodetic datums)
        - DATUM-ENSEMDAT (Ensemble datum)
    - ELLIPSOID (Ellipsoid)
    - PRIMEM (Prime meridian)
    - METHOD (Method / Projection)
    - CS (Coordinate systems)
        - CS-VERTCS (Vertical coordinate system)
        - CS-SPHERCS (Spherical coordinate system)
        - CS-CARTESCS (Cartesian coordinate system)
        - CS-ELLIPCS (Ellipsoidal coordinate system)
        - CS-AFFINE (Affine coordinate system)
        - CS-ORDINAL (Ordinal coordinate system)
    - AXIS (Axis)
    - AREA (Area)
    - UNIT (Unit)
        - UNIT-ANGUNIT (Angle unit)
        - UNIT-SCALEUNIT (Scale unit)
        - UNIT-LENUNIT (Length unit)
        - UNIT-TIMEUNIT (Time unit)
- code
    - Full code of the resource
- trans
    - Code of the transformation
- deprecated
    - \\\\* (Both active and deprecated)
    - 0 (active only - default)
    - 1 (deprecated only)
        transformations: Show detailed transformations for each CRS
        exports: Show exports in WKT and Proj4 notations
        limit: Maximum number of results returned
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/coordinates/search/{query}.json"
    querystring = {}
    if transformations:
        querystring['transformations'] = transformations
    if exports:
        querystring['exports'] = exports
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_geolocation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain information about visitor's location based on IP address of the incoming request."
    
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/geolocation/ip.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def style_json_of_the_map(mapid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Style JSON describing the map cartography. Can be used with various libraries to display a vector map (e.g. Mapbox GL JS, OpenLayers, ...)."
    mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/style.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wmts_capabilities(mapid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "WMTS Capabilities XML document describing the metadata of the map as well as link to the XYZ tiles. Can be used with various GIS software (e.g. QGIS) to display the map."
    mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/WMTSCapabilities.xml"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def embeddable_html_viewer(mapid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Embeddable HTML viewer"
    mapid: Identifier of the map. See [MapTiler Cloud Maps](https://cloud.maptiler.com/maps/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/maps/{mapid}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geojson(dataid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GeoJSON containing the vector features."
    dataid: Identifier of the data. See [MapTiler Cloud Datasets](https://cloud.maptiler.com/data/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/data/{dataid}/features.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tiles_wmts_capabilities(tilesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "WMTS Capabilities XML document describing the metadata of the tiles as well as link to the XYZ tiles. Can be used with various GIS software (e.g. QGIS) to display the tiles."
    tilesid: Identifier of the tiles. See [MapTiler Cloud Tiles](https://cloud.maptiler.com/tiles/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/tiles/{tilesid}/WMTSCapabilities.xml"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_coordinates_reverse(longitude: int, language: str='de,en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reverse geocoding (search by coordinates)."
    language: Prefer results in specific language. One or more [ISO 639-1 codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes), separated by commas to be displayed. Only the first language code is used when prioritizing forward geocode results to be matched. 
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/geocoding/{longitude},{latitude}.json"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def xyz_tiles(x: int, tilesid: str, y: int, z: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The individual tiles. Can be used with various libraries to display the tiles (e.g. Leaflet, OpenLayers, ...). It's usually better (if possible) to use the TileJSON rather than using the tile URL directly."
    tilesid: Identifier of the tiles. See [MapTiler Cloud Tiles](https://cloud.maptiler.com/tiles/).
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/tiles/{tilesid}/{z}/{x}/{y}.{format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_name_forward(query: str, language: str='de,en', bbox: str='5.9559,45.818,10.4921,47.8084', proximity: str='8.528509,47.3774434', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Forward geocoding (search by place name)."
    query: Place name to search for.
        language: Prefer results in specific language. One or more [ISO 639-1 codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes), separated by commas to be displayed. Only the first language code is used when prioritizing forward geocode results to be matched. 
        bbox: A `[w, s, e, n]` bbox array to use for limiting search results. Only features inside the provided bbox will be included.
        proximity: A `[lon, lat]` array to use for biasing search results. Specify to prefer results close to a specific location - features closer to the proximity value will be given priority over those further from the proximity value.
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/geocoding/{query}.json"
    querystring = {}
    if language:
        querystring['language'] = language
    if bbox:
        querystring['bbox'] = bbox
    if proximity:
        querystring['proximity'] = proximity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def font_glyphs(start: int, fontstack: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates the glyphs for the requested fonts. Used when displaying vector maps."
    start: Start of the glyph range.
        fontstack: Font name, or more comma-separated names.
        
    """
    url = f"https://maptiler-cloud-api.p.rapidapi.com/fonts/{fontstack}/{start}-{end}.pbf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptiler-cloud-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

