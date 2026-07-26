from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
import cv2, numpy as np
from ml.pipeline import load_models, run_pipeline
from scripts.make_toy_dataset import make_field

app = FastAPI(title="HemoSight Demo")
_m = {}
def models():
    if not _m: _m["w"], _m["a"] = load_models()
    return _m["w"], _m["a"]

@app.get("/sample")
def sample():
    ok, buf = cv2.imencode(".png", make_field(14, seed=3))
    return Response(content=buf.tobytes(), media_type="image/png")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Could not read image"})
    w, a = models()
    r = run_pipeline(img, w, a)
    return r if isinstance(r, dict) else r.model_dump()

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

PAGE = """
<!doctype html><html><head><meta charset='utf-8'><title>HemoSight</title>
<style>
body{font-family:system-ui,sans-serif;background:#0b0f19;color:#e5e7eb;margin:0;padding:20px}
.banner{background:#7f1d1d;color:#fff;padding:10px;border-radius:6px;font-weight:600;text-align:center;margin-bottom:16px}
h1{color:#f87171}
button,select{font-size:15px;padding:10px 16px;margin:4px;border-radius:6px;border:1px solid #444;background:#1f2937;color:#fff;cursor:pointer}
button:hover{background:#374151}
.row{display:flex;gap:20px;flex-wrap:wrap}
canvas{border:1px solid #444;max-width:420px;background:#000}
.card{background:#111827;border:1px solid #374151;border-radius:8px;padding:12px;margin:6px;min-width:140px;display:inline-block;text-align:center}
.big{font-size:28px;font-weight:700;color:#34d399}
table{border-collapse:collapse;width:100%;margin-top:10px;font-family:monospace;font-size:13px}
td,th{border:1px solid #374151;padding:5px 8px}th{background:#1f2937}
pre{background:#111827;padding:12px;border-radius:8px;overflow:auto;max-height:400px;font-size:12px}
#status{color:#fbbf24;font-weight:600}
</style></head><body>
<div class='banner'>SCREENING AID ONLY - NOT A DIAGNOSTIC DEVICE</div>
<h1>HemoSight - Blood Smear Screening</h1>
<p>1. Load a sample OR choose your own blood-smear image &nbsp; 2. Click Analyze</p>
<input type='file' id='fileInput' accept='image/*'>
<button id='sampleBtn'>Load sample smear</button>
<button id='analyzeBtn'>Analyze</button>
&nbsp; View: <select id='view'><option value='summary'>Summary</option><option value='detailed'>Detailed</option><option value='raw'>Raw JSON</option></select>
<p id='status'></p>
<div class='row'><canvas id='canvas'></canvas><div id='out' style='flex:1;min-width:320px'></div></div>
<img id='preview' style='display:none'>
<script>
let file=null,last=null;
const $=id=>document.getElementById(id);
$('fileInput').onchange=e=>{file=e.target.files[0];preview(file);};
$('sampleBtn').onclick=async()=>{const b=await(await fetch('/sample')).blob();file=new File([b],'sample.png',{type:'image/png'});preview(file);};
$('view').onchange=()=>{if(last)render(last);};
function preview(f){const u=URL.createObjectURL(f);const i=$('preview');i.onload=()=>{const c=$('canvas');c.width=i.naturalWidth;c.height=i.naturalHeight;c.getContext('2d').drawImage(i,0,0);};i.src=u;}
$('analyzeBtn').onclick=async()=>{
  if(!file){alert('Load the sample or choose an image first');return;}
  $('status').textContent='Analyzing...';
  const fd=new FormData();fd.append('file',file);
  const r=await fetch('/analyze',{method:'POST',body:fd});const d=await r.json();last=d;
  $('status').textContent=d.quality_error?'Image rejected (quality)':'Done';
  render(d);boxes(d);
};
function boxes(d){const c=$('canvas');const x=c.getContext('2d');x.lineWidth=2;x.strokeStyle='#f87171';x.fillStyle='#f87171';x.font='12px monospace';(d.detections||[]).forEach(t=>{const[a,b,e,f]=t.bbox;x.strokeRect(a,b,e-a,f-b);x.fillText(t.class_name.slice(0,3),a,b-2);});}
function render(d){
  const v=$('view').value;const o=$('out');
  if(d.quality_error){o.innerHTML='<div class=card style="color:#f87171">Rejected: '+d.quality_error.message+'</div>';return;}
  if(v==='raw'){o.innerHTML='<pre>'+JSON.stringify(d,null,2)+'</pre>';return;}
  const m=d.metrics;
  let h='<div><div class=card>Total cells<div class=big>'+m.total_counts+'</div></div>';
  h+='<div class=card>Malaria flag<div class=big style="color:'+(m.parasite_flag?'#f87171':'#34d399')+'">'+(m.parasite_flag?'FLAGGED':'None')+'</div></div>';
  h+='<div class=card>Needs review<div class=big>'+(d.attention_gate.status==='PASSED'?'No':'Yes')+'</div></div></div>';
  h+='<h3>WBC differential</h3><table><tr><th>Type</th><th>Count</th></tr>';
  for(const k in m.wbc_differential)h+='<tr><td>'+k+'</td><td>'+m.wbc_differential[k]+'</td></tr>';
  h+='</table>';
  if(v==='detailed'){h+='<h3>Detected cells</h3><table><tr><th>#</th><th>Type</th><th>Confidence</th><th>Uncertainty</th></tr>';
    d.detections.forEach(t=>h+='<tr><td>'+t.id+'</td><td>'+t.class_name+'</td><td>'+t.confidence+'</td><td>'+t.uncertainty_std+'</td></tr>');h+='</table>';}
  o.innerHTML=h;
}
</script></body></html>
"""
