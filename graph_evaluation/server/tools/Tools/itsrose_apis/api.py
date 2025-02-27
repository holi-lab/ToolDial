import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_diffusion(sampler: str, ratio: str, controlnet: str, cfg: int, prompt: str, style: str, steps: int, init_image: str=None, negative_prompt: str='((NSFW)), naked, nude, no clothes, upper body uncovered, breast uncovered, (worst quality, low quality, extra digits:1.4),artist name, nsfw, monochrome, fused face, poorly drawn face, cloned face, false body, false face, bad hands, poorly drawn hands, fused eyes, poorly drawn eyes, liquid eyes, false eyes, scary, ugly', seed: int=-1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generate anything from your prompt"
    sampler: Available sampler are

```
Euler a
Euler
LMS
Heun
DPM2
DPM2 a
DPM++ 2S a
DPM++ 2M
DPM++ SDE
DPM fast
DPM adaptive
LMS Karras
DPM2 Karras
DPM2 a Karras
DPM++ 2S a Karras
DPM++ 2M Karras
DPM++ SDE Karras
DDIM
```
        ratio: Available ratio

```
1:1 (512x512)
9:16 (432x768)
16:9 (768x432)
2:3 (512x768)
3:2 (768x512)
```
        controlnet: along with parameter `init_image`

Available controlNet

```
openpose
scribble
hed
canny
```

Default `none`
        cfg: Between 4.0 and 20.0
        prompt: Describe your fantasy in parameter prompt. More detail your prompt more great the image result.
        style: Available style are

```
ACG
Cyberpunk
Realistic
Watercolor
Synthwave
Game CG
Tarot
Ink and Wask
Chibi
Asian
Pastel Colors
```
        steps: Maximum steps `70`
Minimum steps `20`
        init_image: full path image url!
        negative_prompt: describe anything that you don't want appear in image 
        seed: seed
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/diffusion"
    querystring = {'sampler': sampler, 'ratio': ratio, 'controlNet': controlnet, 'cfg': cfg, 'prompt': prompt, 'style': style, 'steps': steps, }
    if init_image:
        querystring['init_image'] = init_image
    if negative_prompt:
        querystring['negative_prompt'] = negative_prompt
    if seed:
        querystring['seed'] = seed
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chatgpt_moderations(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Text filtering"
    text: text that to filter
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/chatGPT/moderations"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_stable_diffusion(prompt: str, ratio: str, cfg: str='7.5', hiresfix: bool=None, negative_prompt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "stable diffusion"
    prompt: describe your fantasy in parameter prompt
        ratio: Available ratio.

> 1:1
> 9:16
> 16:9
> 4:3
> 3:2
> 2:3
> 5:4
> 4:5
> 8:10
> 3:1
> 3:4
        hiresfix: Enable high resolution fix
        negative_prompt: Describe what you don't want appear in image
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/stable/diffusion"
    querystring = {'prompt': prompt, 'ratio': ratio, }
    if cfg:
        querystring['cfg'] = cfg
    if hiresfix:
        querystring['hiresFix'] = hiresfix
    if negative_prompt:
        querystring['negative_prompt'] = negative_prompt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_palette(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "palette"
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/palette"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_arcane(url: str, version: str='v2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "arcane face filter"
    url: full path image url.
        version: All available version
> version = v1, v2, v3.

Default v2.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/arcane"
    querystring = {'url': url, }
    if version:
        querystring['version'] = version
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_yasuo(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "yasuo face filter"
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/yasuo"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_unblur(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "sharpening blurry image"
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/unblur"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_disney(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "disney face filter"
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/disney"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_jojo(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "jojo face filter"
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/jojo"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_drawme(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "face drawing filter"
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/drawMe"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_art(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert face to art style"
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/art"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_aidraw(url: str, prompt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "turn simple drawing image into art"
    url: full path image url.
        prompt: Describe what are you drawing, max char length 20.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/aiDraw"
    querystring = {'url': url, }
    if prompt:
        querystring['prompt'] = prompt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_differentme(url: str, json: bool=None, style: str='anime', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "tunr your face/image to various art style"
    url: full path image url
        json: get json response
        style: list of available style.
```
color_line
fresh
cat_ears
full_bloom
angel
gracefull
cold
snow_fall
manga
charming
stipple
idol
comic_world
anime
princess
```
default `anime`
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/differentMe"
    querystring = {'url': url, }
    if json:
        querystring['json'] = json
    if style:
        querystring['style'] = style
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_comics(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "comics face filter"
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/comics"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_anime_diffusion(prompt: str, negative_prompt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "anime diffusion for weebs"
    prompt: Describe your fantasy in prompt
        negative_prompt: negative prompt.
if `version` parameter set to v2 dont need this parameter.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/anime/diffusion"
    querystring = {'prompt': prompt, }
    if negative_prompt:
        querystring['negative_prompt'] = negative_prompt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_recolor(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    url: full path image url.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/image/recolor"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chatgpt_completions(prompt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "text-davinci*"
    prompt: Your questions, etc.
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/chatGPT/completions"
    querystring = {'prompt': prompt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloader_yt(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "youtube"
    
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/downloader/yt"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloader_ig(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "fetch instagram direct reel/post url"
    
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/downloader/ig"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloader_tiktok(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "fetch  tiktok video w/o watermark and music direct url"
    
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/downloader/tiktok"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloader_fb(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/downloader/fb"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tools_tempmail_new(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "create new temp email"
    name: name for the mail
        
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/tools/tempMail/new"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tools_tempmail_messages(email: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://itsrose-apis1.p.rapidapi.com/tools/tempMail/messages"
    querystring = {}
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "itsrose-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

