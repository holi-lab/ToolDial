import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def doctype_return_a_single_media_type_for_a_file_extension(extension: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON response indicating the media type for the passed in extension and all synonymous extensions this media type supports."
    extension: Query the server whether this file extension is supported and if so return it's assigned media type and alias extensions. Preceding the extension with a dot (.) is optional, but is allowed if including it is more convenient.

If this query param isn't present or is set to an asterisk, all supported media types and their extensions are returned.
        
    """
    url = f"https://tweekit.p.rapidapi.com/docType"
    querystring = {'extension': extension, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tweekit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def doctype_return_all_supported_media_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a JSON response array with all supported media types and their extensions. The only difference between this endpoint and the endpoint which returns the media type for a specific extension is the "extension" query param is not specified, causing all media types to be returned."
    
    """
    url = f"https://tweekit.p.rapidapi.com/docType"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tweekit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_final_image(docid: str, page: int=1, height: int=240, x2: int=0, download: bool=None, fmt: str=None, y2: int=0, bg: str='0xFFFFFF', alpha: bool=None, y1: int=0, x1: int=0, width: int=320, elliptical: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate the final image in a specified file format from the original document using the given TweekIt parameters. After this call completes successfully, the original document is permanently deleted from our server."
    docid: Use the docId returned from the upload endpoint to reference the uploaded original document. if testing this from the RapidAPI UI, copy the docId from the upload endpoint's response data and paste it in here.
        page: Return a specific page or frame from multipage documents or multiframe images. If not present, it defaults to 1.

Currently, specifying multiple pages or a page range is not supported.
        height: The height of the output image, in pixels. The image is scaled if necessary. If not present, the height is based on the height of the specified crop box. If neither the height nor a crop is specified, it defaults to the height of the original document.

If the height is specified but not the width, the output width is proportionally adjusted to maintain the aspect ratio. If both a width and height are specified, the image is padded with the background color or with a transparent alpha if possible and alpha mode is enabled.
        x2: Right side coordinate of the crop box, in pixels. Defaults to the same value of x1 which indicates no right side cropping. This value can also extend past the pixel edge of the original document which causes the right side to be padded with transparency or the background color.

This value must be at least one greater than x1 for horizontal cropping to occur. If they are identical or x2 < x1, horizontal cropping is disabled.

Cropping is applied before scaling by the output width and height, if applicable.
        download: Causes a content-disposition response header to be inserted into the response, triggering the browser to display a download dialog window. Defaults to false.

If this API is not being used in a browser, the parameter will have no effect. Also, RapidAPI may not support passing this response through. Contact TweekIT support for direct access to the API if downloading of output images is required.
        fmt: Specifies the format of the generated output image or document. If not present, it defaults to JPEG.

Currently, multi-page/multi-frame output for PDF and GIF is not supported. Only a single page or frame can be output at this time.

Currently, all metadata present in the original document is stripped from the generated image for privacy reasons. Support for passing through and/or inserting metadata may come at a later date.
        y2: Bottom coordinate of the crop box, in pixels. Defaults to the same value of y1 which indicates no bottom cropping. This value can also extend past the pixel edge of the original document which causes the bottom to be padded with transparency or the background color.

This value must be at least one greater than y1 for vertical cropping to occur. If they are identical or y2 < y1, vertical cropping is disabled.

Cropping is applied before scaling by the width and height, if applicable.
        bg: The output image background color. The value can be a decimal integer, hex value, or web color formatted value. Defaults to black (0) if not present.

If alpha is disabled or the output format does not support an alpha (or a transparent color in the case of GIF) then any transparent areas in the original document are replaced by this background color.

If a width and height were both specified and the original image doesn't match the aspect ratio of the output image, the output image is padded with this background color if alpha is disabled or an alpha cannot be used.
        alpha: Enables/disables the alpha channel for 32-bit output formats or transparent pixel colors in GIF. Defaults to unset, which causes alpha handling to be based on whether the original document contains an alpha channel and whether the format of the output image format supports transparency.

Also, the elliptical parameter affects this parameter if it is unset. Enabling elliptical output causes alpha to be enabled if the output format supports it, even if the original document does not contain an alpha channel.

If set to true, an alpha channel is generated/copied to the output image if possible. Formats like JPEG and PDF do not support an alpha channel so setting this parameter to true will have no effect.

if set to false, all alpha channels are replaced with the background color, even if the output format supports an alpha or transparent pixels.
        y1: Top coordinate of the crop box, in pixels. Defaults to 0 which indicates no top cropping. This value can also be negative which causes the top of the output image to be padded with transparency or the background color.
        x1: Left side coordinate of the crop box, in pixels. Defaults to 0 which indicates no left side cropping. This value can also be negative which causes the left side to be padded with transparency or the background color.
        width: The width of the output image, in pixels. The image is scaled if necessary. If not present, the width is based on the width of the specified crop box. If neither the width nor a crop is specified, it defaults to the width of the original document.

If the width is specified but not the height, the output height is proportionally adjusted to maintain the aspect ratio. If both a width and height are specified, the image is padded with the background color or with a transparent alpha if possible and alpha mode is enabled.
        elliptical: Enables/disables elliptical cropping. Defaults to false. If cropping is disabled, this parameter is ignored.

The areas outside the ellipse but inside the crop box are filled with an empty alpha channel (if alpha can be used or is enabled) or with the background color. The edge of the ellipse is anti-aliased.
        
    """
    url = f"https://tweekit.p.rapidapi.com/{docid}"
    querystring = {}
    if page:
        querystring['page'] = page
    if height:
        querystring['height'] = height
    if x2:
        querystring['x2'] = x2
    if download:
        querystring['download'] = download
    if fmt:
        querystring['fmt'] = fmt
    if y2:
        querystring['y2'] = y2
    if bg:
        querystring['bg'] = bg
    if alpha:
        querystring['alpha'] = alpha
    if y1:
        querystring['y1'] = y1
    if x1:
        querystring['x1'] = x1
    if width:
        querystring['width'] = width
    if elliptical:
        querystring['elliptical'] = elliptical
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tweekit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def preview_generate_preview_image(docid: str, page: int=1, height: int=240, y1: int=0, alpha: bool=None, fmt: str=None, width: int=320, x1: int=0, y2: int=0, elliptical: bool=None, x2: int=0, download: bool=None, bg: str='0xFFFFFF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a preview image in a specified file format from the original document using the given TweekIt parameters."
    docid: Use the docId returned from the upload endpoint to reference the uploaded original document. if testing this from the RapidAPI UI, copy the docId from the upload endpoint's response data and paste it in here.
        page: Return a specific page or frame from multipage documents or multiframe images. If not present, it defaults to 1.

Currently, specifying multiple pages or a page range is not supported.
        height: The height of the output image, in pixels. The image is scaled if necessary. If not present, the height is based on the height of the specified crop box. If neither the height nor a crop is specified, it defaults to the height of the original document.

If the height is specified but not the width, the output width is proportionally adjusted to maintain the aspect ratio. If both a width and height are specified, the image is padded with the background color or with a transparent alpha if possible and alpha mode is enabled.
        y1: Top coordinate of the crop box, in pixels. Defaults to 0 which indicates no top cropping. This value can also be negative which causes the top of the output image to be padded with transparency or the background color.
        alpha: Enables/disables the alpha channel for 32-bit output formats or transparent pixel colors in GIF. Defaults to unset, which causes alpha handling to be based on whether the original document contains an alpha channel and whether the format of the output image format supports transparency.

Also, the elliptical parameter affects this parameter if it is unset. Enabling elliptical output causes alpha to be enabled if the output format supports it, even if the original document does not contain an alpha channel.

If set to true, an alpha channel is generated/copied to the output image if possible. Formats like JPEG and PDF do not support an alpha channel so setting this parameter to true will have no effect.

if set to false, all alpha channels are replaced with the background color, even if the output format supports an alpha or transparent pixels.
        fmt: Specifies the format of the generated output image or document. If not present, it defaults to JPEG.

Currently, multi-page/multi-frame output for PDF and GIF is not supported. Only a single page or frame can be output at this time.

Currently, all metadata present in the original document is stripped from the generated image for privacy reasons. Support for passing through and/or inserting metadata may come at a later date.
        width: The width of the output image, in pixels. The image is scaled if necessary. If not present, the width is based on the width of the specified crop box. If neither the width nor a crop is specified, it defaults to the width of the original document.

If the width is specified but not the height, the output height is proportionally adjusted to maintain the aspect ratio. If both a width and height are specified, the image is padded with the background color or with a transparent alpha if possible and alpha mode is enabled.
        x1: Left side coordinate of the crop box, in pixels. Defaults to 0 which indicates no left side cropping. This value can also be negative which causes the left side to be padded with transparency or the background color.
        y2: Bottom coordinate of the crop box, in pixels. Defaults to the same value of y1 which indicates no bottom cropping. This value can also extend past the pixel edge of the original document which causes the bottom to be padded with transparency or the background color.

This value must be at least one greater than y1 for vertical cropping to occur. If they are identical or y2 < y1, vertical cropping is disabled.

Cropping is applied before scaling by the width and height, if applicable.
        elliptical: Enables/disables elliptical cropping. Defaults to false. If cropping is disabled, this parameter is ignored.

The areas outside the ellipse but inside the crop box are filled with an empty alpha channel (if alpha can be used or is enabled) or with the background color. The edge of the ellipse is anti-aliased.
        x2: Right side coordinate of the crop box, in pixels. Defaults to the same value of x1 which indicates no right side cropping. This value can also extend past the pixel edge of the original document which causes the right side to be padded with transparency or the background color.

This value must be at least one greater than x1 for horizontal cropping to occur. If they are identical or x2 < x1, horizontal cropping is disabled.

Cropping is applied before scaling by the output width and height, if applicable.
        download: Causes a content-disposition response header to be inserted into the response, triggering the browser to display a download dialog window. Defaults to false.

If this API is not being used in a browser, the parameter will have no effect. Also, RapidAPI may not support passing this response through. Contact TweekIT support for direct access to the API if downloading of output images is required.
        bg: The output image background color. The value can be a decimal integer, hex value, or web color formatted value. Defaults to black (0) if not present.

If alpha is disabled or the output format does not support an alpha (or a transparent color in the case of GIF) then any transparent areas in the original document are replaced by this background color.

If a width and height were both specified and the original image doesn't match the aspect ratio of the output image, the output image is padded with this background color if alpha is disabled or an alpha cannot be used.
        
    """
    url = f"https://tweekit.p.rapidapi.com/preview/{docid}"
    querystring = {}
    if page:
        querystring['page'] = page
    if height:
        querystring['height'] = height
    if y1:
        querystring['y1'] = y1
    if alpha:
        querystring['alpha'] = alpha
    if fmt:
        querystring['fmt'] = fmt
    if width:
        querystring['width'] = width
    if x1:
        querystring['x1'] = x1
    if y2:
        querystring['y2'] = y2
    if elliptical:
        querystring['elliptical'] = elliptical
    if x2:
        querystring['x2'] = x2
    if download:
        querystring['download'] = download
    if bg:
        querystring['bg'] = bg
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tweekit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

