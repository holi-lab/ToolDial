import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def kiwiki_boutique(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Kiwiki Boutique là cửa hàng đăng và bán sản phẩm đến từ các thương hiệu nổi tiếng chính hãng trên toàn thế giới. Cụ thể các sản phẩm mà Kiwiki Boutique cung cấp đó là: túi xách hàng hiệu, đồng hồ, gọng kính, khăn quàng, phụ kiện hàng hiệu, các loại sản phẩm hàng hiệu secondhand, hàng hiệu chính hãng đã qua sử dụng có chất lượng tốt và giá cả phù hợp.website đều được mô tả 1 cách chân thực nhất
		.#hanghieuchinhhang #hanghieusecondhand #KiwikiBoutique #nuochoasecondhand #tuisecondhand #giaydepsecondhand #donghosecondhand #kinhmatsecondhand #khansecondhand#second hand #2hand #authenticsecond hand #authenticdaquasu dung #hanghieucu #vintagecu
		Địa chỉ : 54 - Phố Tam Khương, Khương Thượng, Đống Đa, Hà Nội
		Hotline 0981987140
		Gmail: kiwiki911@gmail.com
		Google map https://www.google.com/maps?cid=148938482515809140    
		Website https://kiwiki.vn/ 
		https://kiwiki.vn/dong-ho/ 
		https://kiwiki.vn/tui-xach/ 
		https://kiwiki.vn/kinh-mat/ 
		https://kiwiki.vn/vi-nu-nam/ 
		https://www.facebook.com/kiwikiboutique/ 
		https://experiment.com/users/kiwikibouti_que 
		https://comicvine.gamespot.com/profile/kiwikiboutique/
		https://gifyu.com/kiwikiboutique
		https://folkd.com/user/kiwikiboutique
		https://sketchfab.com/kiwikiboutique
		http://kiwikiboutique.wikidot.com/main:about
		https://gotartwork.com/Profile/kiwiki-boutique/146211/
		https://stocktwits.com/kiwikiboutique
		https://connect.garmin.com/modern/profile/51bb57e2-2f4f-4091-9202-5c02a6041d0e
		https://ko-fi.com/kiwikiboutique
		https://linktr.ee/kiwikiboutique
		https://onlyfans.com/kiwikiboutique 
		https://gallery.autodesk.com/users/HBW2TSGGNKGPGAYG?relProf=1
		https://gitlab.manjaro.org/kiwikiboutique
		https://gitlab.pasteur.fr/kiwikiboutique
		https://repo.getmonero.org/kiwikiboutique
		https://career.habr.com/kiwikiboutique
		https://gitlab.haskell.org/kiwikiboutique
		https://cartoonmovement.com/cartoonist/18537
		https://www.pinterest.com/kiwikiboutiquevn/_saved/
		https://www.behance.net/kiwikiboutique
		https://vi.gravatar.com/kiwikiboutique
		https://www.jigsawplanet.com/kiwikiboutique?viewas=0436d5029913
		https://www.divephotoguide.com/user/kiwikiboutique
		https://fr.ulule.com/kiwikiboutiquevn/#/
		https://my.archdaily.com/us/@kiwikiboutiquevn
		http://codepad.org/users/kiwikiboutique
		https://www.twitch.tv/kiwikiboutique/about"
    
    """
    url = f"https://kiwiki-boutique.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kiwiki-boutique.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

