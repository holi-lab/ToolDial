import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def winart(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "WINART là nhà cung cấp các giải pháp thi công lắp đặt hàng đầu về mành rèm tại thị trường Hà Nội nói riêng và Việt Nam nói chung. WinArt mang đến cho bạn một thế giới mành rèm với đa dạng kiểu dáng, mẫu mã và những công năng ưu việt.Dây chuyền và nguyên vật liệu sản xuất đồng bộ được chuyển giao từ Tập đoàn hàng đầu Hàn Quốc giúp gia tăng chất lượng, đảm bảo độ chính xác và nâng cao chất lượng sản phẩm.
		#rèm_cầu_vồng #rèm_cửa_Hàn Quốc #Rèm_cuốn #Rèm_vải #Hà_Nội #winart
		Địa chỉ : 206 Phố Vọng Phương Liệt Thanh Xuân Hà Nội
		Hotline 090 922 62 29
		Email: winartvn@gmail.com
		Google map https://www.google.com/maps?cid=1256227693022228604 
		Website https://winart.vn/ 
		https://winart.vn/rem-cau-vong/ 
		https://winart.vn/rem-cuon/ 
		https://winart.vn/rem-vai/ 
		https://www.pinterest.com/winartvn/
		https://www.behance.net/winartvn
		https://folkd.com/user/winart
		https://www.diigo.com/user/winartvn
		https://visual.ly/users/winartvn/portfolio
		http://wiki.cs.hse.ru/Обсуждение_участника:Winart 
		https://www.goodreads.com/user/show/149834613-win-art
		https://lab.quickbox.io/winart
		https://hypothes.is/users/winart
		https://my.desktopnexus.com/winart/
		https://www.allmyfaves.com/winart
		https://www.giantbomb.com/profile/winart/
		https://zoimas.com//profile/winart/about
		https://medium.com/@winartvn/about
		https://www.podomatic.com/podcasts/winartvn
		https://gitlab.tue.nl/winart
		https://www.magcloud.com/user/winart
		https://forum.cs-cart.com/user/201665-winart/
		https://www.beatstars.com/winart
		https://www.ranker.com/writer/winartvn
		https://repo.getmonero.org/winart
		https://replit.com/@winartvn
		https://leetcode.com/winartvn/
		https://platform.xr4all.eu/winartvn
		https://www.emoneyspace.com/winart
		https://www.atlasroleplay.com/forum/profile/winart
		https://rapidapi.com/user/winart
		https://lookbook.nu/winartvn
		https://www.quia.com/profiles/winar302
		https://www.teachertube.com/user/channel/winart
		https://www.madinamerica.com/forums/users/winart/
		https://triberr.com/winartvn
		https://www.catchafire.org/profiles/2110859/
		https://miarroba.com/winart
		https://writeablog.net/winart/winart
		https://itsmyurls.com/winartvn
		https://mastodon.online/@winart
		https://forums.bohemia.net/profile/1180323-winartvn/?tab=field_core_pfield_141
		https://www.themesindep.com/support/forums/users/winartvn/
		https://forum.magicmirror.builders/user/winart
		https://bitcoinblack.net/community/winart/info/
		https://www.bitchute.com/channel/afdMVqxFKsE2/
		https://exercism.org/profiles/winart
		https://hearthis.at/winart/set/winart/
		https://gitlab-test.eclipse.org/-/snippets/124
		https://www.dwell.com/collection/6917732180818931712
		https://bookme.name/winart
		https://twitter.com/winartvn
		https://www.linkedin.com/in/winartvn/"
    
    """
    url = f"https://winart.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "winart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

