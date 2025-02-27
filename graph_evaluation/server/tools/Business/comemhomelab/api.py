import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def comemhomelab(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cỏ Mềm Homelab là thương hiệu mỹ phẩm thiên nhiên lành và thật được chiết xuất từ thảo mộc an toàn với con người đặc biệt là bà bầu, trẻ em và da kích ứng
		#myphamthiennhien #myphambabau #daugoithaoduoc #soncomem# nuochoakho #kemchongnangthiennhien #SerumDuongDaBanDem #muoitamcafe #Serumduongtoc #daugoitrigauthaoduoc #daugoinganrungtoc
		Địa chỉ 225 Trần Đăng Ninh, Dịch Vọng, Cầu Giấy, Hà Nội 
		Hotline 1800646890 |0934849339 | NTL : 25/11/2015
		EmaiL: cskh@comem.vn
		Google map https://www.google.com/maps?cid=17541894170222349188 
		Website https://comem.vn/ 
		https://comem.vn/cm/son-moi 
		https://comem.vn/cm/nuoc-hoa 
		https://comem.vn/cm/kem-chong-nang 
		https://www.facebook.com/comemhomelab/ 
		https://gitlab.pasteur.fr/comemhomelab
		https://git.qoto.org/comemhomelab 
		https://folkd.com/user/comemhomelab 
		https://www.gametabs.net/user/420436
		https://comemvn.contently.com/ 
		https://comicvine.gamespot.com/profile/comemhomelab/ 
		https://www.giantbomb.com/profile/comemhomelab/ 
		https://twinoid.com/user/10046568
		https://platform.xr4all.eu/comemvn1 
		https://www.producthunt.com/@comemhomelab
		https://support.wedesignthemes.com/users/comemhomelab/
		https://descubre.beqbe.com/p/comemhomelab
		https://ko-fi.com/comemhomelab
		https://easypropertylistings.com.au/support/users/comemhomelab/
		https://padlet.com/comemhomelab
		https://linktr.ee/comemhomelab 
		https://www.mojomarketplace.com/user/comemhomelab-t04xVd27DA
		https://dev.to/comemhomelab
		https://www.walkscore.com/people/281000336138/comemhomelab
		https://pastebin.com/u/comemhomelab1
		https://camp-fire.jp/profile/comemhomelab
		https://community.opengroup.org/comemhomelab
		https://www.360cities.net/profile/comemhomelab
		https://www.diggerslist.com/comemhomelab/about
		https://www.credly.com/users/comem-homelab/badges
		https://marketplace.apartmenttherapy.com/store/56c8f836-0a19-4c00-8339-14797093225c/about
		https://www.diigo.com/profile/comemhomelab
		https://www.scoop.it/u/comemhomelab
		https://qiita.com/comemhomelab1
		https://band.us/band/87562397 
		https://www.elephantjournal.com/profile/comemvn1/"
    
    """
    url = f"https://comemhomelab.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "comemhomelab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

