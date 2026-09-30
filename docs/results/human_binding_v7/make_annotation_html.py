"""生成**单文件、全中文**的标注页（发给填表人只需这一个文件）。

要满足的三点（用户 2026-09-23 要求）：
  1. 提示词给中文翻译（填表人只看得懂中文）；
  2. 填写说明写在页面里，不靠外部 README；
  3. 只有一个文件：图片 base64 内嵌，双击即可用，不联网、不装东西。

选项给四个：绑定正确 / 属性绑错 / 对象缺失 / 无法判断（"无法判断"不计入统计）。
导出 CSV 可直接喂给 score_annotation.py。
"""
import base64
import csv
import html
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, '/tmp/cb_probe')
from prompt_zh import ZH

D = Path('/root/private_data/PixArt-alpha-attentiongate/results/human_binding_v7')
SHEET = D / 'annotation_sheet.csv'
OUT = D / '发给标注者' / '标注页.html'
THUMB, QUALITY = 300, 84

OPTIONS = [('', '— 请选择 —'),
           ('proper', '① 绑定正确'),
           ('improper', '② 属性绑错'),
           ('neglect', '③ 对象缺失'),
           ('unsure', '④ 无法判断')]


def b64(path):
    im = Image.open(path).convert('RGB').resize((THUMB, THUMB), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def opts():
    return ''.join(f'<option value="{v}">{html.escape(t)}</option>'
                   for v, t in OPTIONS)


def main():
    rows = list(csv.DictReader(SHEET.open()))
    parts = []
    for r in rows:
        iid = int(r['item_id'])
        zh = ZH.get(iid, '')
        parts.append(f'''<tr id="r{iid}">
  <td class="id">#{iid}</td>
  <td class="prompt"><div class="en">{html.escape(r['prompt'])}</div>
      <div class="zh">{html.escape(zh)}</div></td>
  <td><img src="data:image/jpeg;base64,{b64(r['image_A'])}" alt="图A"></td>
  <td><img src="data:image/jpeg;base64,{b64(r['image_B'])}" alt="图B"></td>
  <td><select data-i="{iid}" data-s="A">{opts()}</select></td>
  <td><select data-i="{iid}" data-s="B">{opts()}</select></td>
</tr>''')

    # JS 单独写成普通字符串再拼进去——放在 f-string 里会把所有大括号当占位符
    JS = r"""
var KEY='binding_anno_zh_v7', MEM={}, STORAGE_OK=true;
function read(){ try{ return JSON.parse(localStorage.getItem(KEY)||'{}'); }
                 catch(e){ STORAGE_OK=false; return MEM; } }
function write(o){ try{ localStorage.setItem(KEY,JSON.stringify(o)); }
                   catch(e){ STORAGE_OK=false; MEM=o; } }
var sels=document.querySelectorAll('select');
function keyOf(el){ return el.getAttribute('data-i')+'-'+el.getAttribute('data-s'); }
(function init(){
  var saved=read();
  for(var i=0;i<sels.length;i++){
    var s=sels[i], k=keyOf(s);
    if(saved[k]!==undefined) s.value=saved[k];
    s.onchange=function(){ var o=read(); o[keyOf(this)]=this.value; write(o); mark(); prog(); };
  }
  mark(); prog();
  if(!STORAGE_OK){
    var w=document.getElementById('warn'); w.style.display='block';
    w.textContent='提示：这个浏览器禁用了本地存储，填写内容不会自动保存。'
      +'请一次填完后立刻点「显示结果文本」把内容复制走。';
  }
})();
function mark(){
  var o=read(), trs=document.querySelectorAll('tbody tr');
  for(var i=0;i<trs.length;i++){
    var tr=trs[i], id=tr.id.slice(1), a=o[id+'-A'], b=o[id+'-B'];
    tr.className=(a&&b)?'done':((a||b)?'half':'');
  }
}
function prog(){
  var n=0, done=0, trs=document.querySelectorAll('tbody tr');
  for(var i=0;i<sels.length;i++){ if(sels[i].value) n++; }
  for(var j=0;j<trs.length;j++){ if(trs[j].className==='done') done++; }
  document.getElementById('prog').textContent =
    '已完成 '+done+' / __NROWS__ 行（共 '+n+' / '+sels.length+' 个判定）';
}
function csvText(){
  var out='item_id,annotator1,annotator2,notes\r\n';
  var trs=document.querySelectorAll('tbody tr');
  for(var t=0;t<trs.length;t++){
    var tr=trs[t], id=tr.id.slice(1);
    var sa=tr.querySelector('select[data-s="A"]').value;
    var sb=tr.querySelector('select[data-s="B"]').value;
    var cell=(sa||sb)?('A:'+sa+' B:'+sb):'';
    out+=id+',"'+cell+'",,\r\n';
  }
  return out;
}
function showText(msg){
  var txt;
  try{ txt=csvText(); }catch(e){ alert('读取填写内容失败：'+e); return; }
  document.getElementById('outbox').style.display='block';
  document.getElementById('outmsg').textContent=msg;
  var ta=document.getElementById('outta');
  ta.value=txt; ta.focus(); ta.select();
  try{ document.execCommand('copy'); }catch(e){}
}
function exportCSV(){
  var txt;
  try{ txt=csvText(); }catch(e){ alert('读取填写内容失败：'+e); return; }
  var ok=false;
  try{
    var blob=new Blob(['\ufeff'+txt],{type:'text/csv;charset=utf-8'});
    var url=URL.createObjectURL(blob);
    var a=document.createElement('a');
    a.href=url; a.download='annotation_filled.csv';
    document.body.appendChild(a); a.click();
    setTimeout(function(){ document.body.removeChild(a); URL.revokeObjectURL(url); },1500);
    ok=true;
  }catch(e){ ok=false; }
  showText(ok? '已尝试下载 annotation_filled.csv。若下载文件夹里没有，请直接复制下面框里的内容发回。'
             : '这个浏览器不允许自动下载，请直接复制下面框里的内容发回。');
}
""".replace('__NROWS__', str(len(rows)))

    page = f'''<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>图片属性绑定标注（200 行）</title>
<style>
 body{{font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
       margin:0;padding:0 16px 40px;background:#f6f7f9;color:#222}}
 h1{{font-size:19px;margin:14px 0 8px}}
 .howto{{background:#fff;border:1px solid #d8dee6;border-left:5px solid #2f6fb5;
        border-radius:6px;padding:12px 16px;margin-bottom:12px;line-height:1.75;font-size:14px}}
 .howto b{{color:#1f6fb5}}
 .howto table{{border-collapse:collapse;margin:6px 0}}
 .howto td{{border:1px solid #dfe5ec;padding:3px 10px;font-size:13.5px}}
 .bar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
       padding:10px 0;margin-bottom:10px;z-index:5;display:flex;align-items:center;gap:12px}}
 #prog{{font-size:14px;color:#1f6fb5;font-weight:bold}}
 button{{font-size:14px;padding:7px 16px;cursor:pointer;border-radius:4px;
        border:1px solid #2f6fb5;background:#2f6fb5;color:#fff}}
 button.gray{{background:#fff;color:#444;border-color:#bbb}}
 table.main{{border-collapse:collapse;background:#fff;width:100%}}
 table.main th{{position:sticky;top:56px;background:#eef2f7;font-size:13.5px;
       padding:8px 6px;border:1px solid #dde3ea}}
 table.main td{{border:1px solid #e6e6e6;padding:6px;vertical-align:middle}}
 td.id{{color:#666;width:46px;text-align:right}}
 td.prompt{{width:320px;line-height:1.45}}
 .en{{font-size:13px;color:#333}}
 .zh{{font-size:14px;color:#111;margin-top:3px}}
 td img{{display:block;width:{THUMB}px;height:{THUMB}px;border-radius:3px}}
 select{{font-size:14px;padding:5px;min-width:118px}}
 tr.done{{background:#f3fbf5}}
 tr.half{{background:#fffaf0}}
</style></head><body>

<h1>图片属性绑定标注</h1>

<div class="howto">
  <b>要做什么</b>：每一行给出一条提示词和两张图（<b>左=图A，右=图B</b>）。
  请判断每张图有没有把提示词里说的<b>颜色</b>正确安到<b>对应的物体</b>上。
  <b>图A 和图B 各判一次</b>，两张图互不影响（不是"改动前后"，请独立判断）。<br>
  <b>怎么选</b>：
  <table>
    <tr><td>① 绑定正确</td><td>提示词里的物体都出现了，而且颜色都安在正确的物体上</td></tr>
    <tr><td>② 属性绑错</td><td>物体都出现了，但至少有一个颜色安到了错误的物体上（该蓝的变红、两个物体颜色互换等）</td></tr>
    <tr><td>③ 对象缺失</td><td>提示词里的某个物体<b>根本没画出来</b>（不论颜色对不对）</td></tr>
    <tr><td>④ 无法判断</td><td>看不清、或几种情况混杂难以归类（不计入统计，但请尽量选①②③）</td></tr>
  </table>
  <b>填完怎么办</b>：点「显示结果文本」（或「导出结果」），把框里的内容复制发回即可。
  填的内容会自动保存在这台电脑的浏览器里；若浏览器禁用了本地存储，页面顶部会提示你。
</div>

<div id="warn" style="display:none;background:#fff4e5;border:1px solid #e0a458;
     border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:14px"></div>

<div class="bar">
  <button onclick="exportCSV()">导出结果</button>
  <button class="gray" onclick="showText('已把结果放进下面的框里（已尝试复制到剪贴板）。')">显示结果文本</button>
  <button class="gray" onclick="if(confirm('清空本页所有已填内容？')){{localStorage.removeItem(KEY);location.reload()}}">清空重来</button>
  <span id="prog"></span>
</div>

<div id="outbox" style="display:none;background:#fff;border:1px solid #2f6fb5;
     border-radius:6px;padding:12px;margin-bottom:12px">
  <div id="outmsg" style="font-size:14px;color:#1f6fb5;margin-bottom:6px"></div>
  <textarea id="outta" style="width:100%;height:150px;font-family:monospace;font-size:12px"></textarea>
</div>

<table class="main">
<thead><tr><th>#</th><th>提示词（上：英文原文 / 下：中文）</th>
<th>图 A</th><th>图 B</th><th>图A 判定</th><th>图B 判定</th></tr></thead>
<tbody>
{''.join(parts)}
</tbody></table>

<script>{JS}</script></body></html>'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding='utf-8')
    print(f'-> {OUT}')
    print(f'   {len(rows)} 行，单文件 {OUT.stat().st_size/1048576:.1f} MB，含中文翻译与页内说明')


if __name__ == '__main__':
    main()
