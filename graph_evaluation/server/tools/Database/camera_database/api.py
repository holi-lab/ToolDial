import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lenses(focal_length: int=None, maximum_aperture: int=None, page_size: int=None, aperture_ring: bool=None, filter_thread: int=None, groups: int=None, brand: str='canon', autofocus: bool=None, page: int=None, minimum_aperture: int=None, name: str=None, elements: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API gets lenses."
    focal_length: Focal length. If it's a variable aperture, e.g. 18-50, 18 or 50 would both match.
        maximum_aperture: Maximum Aperture.
        page_size: Page size, default is 10. Max is 100.
        aperture_ring: Indicates whether the lens has an aperture ring.
        filter_thread: Filter thread in mm.
        groups: Amount of groups.
        brand: Brand of the camera
        autofocus: Indicates whether the lens has autofocus.
        page: Page to be querried. Default is the first page.
        minimum_aperture: Minimum aperture.
        name: Name of the lens. Case insensitive and wild match.
        elements: Number of elements in a lens
        
    """
    url = f"https://camera-database.p.rapidapi.com/lenses"
    querystring = {}
    if focal_length:
        querystring['focal_length'] = focal_length
    if maximum_aperture:
        querystring['maximum_aperture'] = maximum_aperture
    if page_size:
        querystring['page_size'] = page_size
    if aperture_ring:
        querystring['aperture_ring'] = aperture_ring
    if filter_thread:
        querystring['filter_thread'] = filter_thread
    if groups:
        querystring['groups'] = groups
    if brand:
        querystring['brand'] = brand
    if autofocus:
        querystring['autofocus'] = autofocus
    if page:
        querystring['page'] = page
    if minimum_aperture:
        querystring['minimum_aperture'] = minimum_aperture
    if name:
        querystring['name'] = name
    if elements:
        querystring['elements'] = elements
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "camera-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cameras(megapixels: int=None, page_size: int=None, brand: str='canon', sensor_size: str='full_frame', name: str=None, announced: int=None, page: int=None, lens_mount: str='canon_ef_efs', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API gets cameras."
    megapixels: Effective megapixels of the camera
        page_size: Page size, default is 10. Max is 100.
        brand: Brand of the camera
        sensor_size: Sensor size.
        name: Name of the camera. Case insensitive.
        announced: Announced year. The database has cameras from 1996.
        page: Page to be querried. Default is the first page.
        lens_mount: Lens mount, or brand.
        
    """
    url = f"https://camera-database.p.rapidapi.com/cameras"
    querystring = {}
    if megapixels:
        querystring['megapixels'] = megapixels
    if page_size:
        querystring['page_size'] = page_size
    if brand:
        querystring['brand'] = brand
    if sensor_size:
        querystring['sensor_size'] = sensor_size
    if name:
        querystring['name'] = name
    if announced:
        querystring['announced'] = announced
    if page:
        querystring['page'] = page
    if lens_mount:
        querystring['lens_mount'] = lens_mount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "camera-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

