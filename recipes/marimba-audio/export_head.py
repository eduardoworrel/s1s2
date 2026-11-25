"""Exporta a cabeça de fala (voz ativa) do Marimba para ONNX e converte as leituras lineares/MLP (turno, locutor) para JSON float32.
Entrada da cabeça: h9 nativo [104,896] (saída do encoder ONNX), máscara de quadros [100] (float 0/1), question_pool [1,256] (pré-calculado do prompt canônico).
Dentro do grafo: interpolação linear 104->100 (align_corners=False), ida e volta fp16, projeção, 2 camadas condicionadas (FiLM), atenção da opção "sim", score."""
import json,base64,numpy as np,torch,torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from safetensors.numpy import load_file as load_np
from transformers import WhisperTokenizer
from marimba._head import SpeechModel,GraphemeTokenizer,TEXT_REPO,TEXT_REVISION,CachedQuestionTokenizer
from marimba.pretrained import _render_vad_prompt,_CANONICAL_VAD_STEM,_VAD_TEMPERATURE,_VAD_TAIL_TEMPERATURE
S=sorted(Path.home().glob(".cache/huggingface/hub/models--charcot-ai--marimba/snapshots/*"))[0]
torch.backends.mha.set_fastpath_enabled(False)
m=SpeechModel(); m.load_state_dict(load_file(str(S/"speech_head/model.safetensors")),strict=True); m.eval().requires_grad_(False)
tok=CachedQuestionTokenizer(WhisperTokenizer.from_pretrained(TEXT_REPO,revision=TEXT_REVISION))
ids,mask=GraphemeTokenizer()("sim")
with torch.no_grad(): option=m.text_options(ids,mask)
prompt=_render_vad_prompt(_CANONICAL_VAD_STEM)
enc=tok([prompt]); qi=enc["input_ids"]; qm=enc["attention_mask"].bool()
with torch.no_grad(): q=m.text(qi,qm)
qpool=m.head.pool(q,qm)   # [1,256]
print("prompt tokens",qi.shape[1],"qpool",tuple(qpool.shape))
class Head(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.h=m.head; s.register_buffer("opt",option); s.register_buffer("optm",mask)
    def forward(s,h9,fmask,qpool):
        x=F.interpolate(h9.T.unsqueeze(0),size=100,mode="linear",align_corners=False).transpose(1,2).half().float()
        mk=fmask>0.5
        h=s.h; mem=h.projection_audio(x)
        for L in h.layers_audio:
            mem=L.transformer(mem,src_key_padding_mask=~mk)
            g,b=L.film(qpool).chunk(2,dim=-1); mem=mem*(1+g.unsqueeze(1))+b.unsqueeze(1)
        mem=h.normalized_audio(mem)
        ctx,_=h.attention(s.opt,mem,mem,key_padding_mask=~mk,need_weights=False)
        return h.score(h.pool(s.opt+ctx,s.optm)).reshape(-1)
w=Head().eval()
ref=np.load("ref/_dummy.npy") if False else None
rng=np.random.default_rng(0); x=torch.tensor(rng.normal(0,1,(104,896)),dtype=torch.float32); fm=torch.ones(1,100)
fm[0,:30]=0
with torch.no_grad(): o=w(x,fm,qpool)
# referência: caminho original do pacote (score de SpeechReadout)
with torch.no_grad():
    fr=F.interpolate(x.T.unsqueeze(0),size=100,mode="linear",align_corners=False).transpose(1,2).half().float()
    o2=m.head(fr,fm.bool(),q,qm,option,mask)
print("wrapper vs original",float(o),float(o2))
out="vad_head.onnx"
torch.onnx.export(w,(x,fm,qpool),out,input_names=["h9","fmask","qpool"],output_names=["score"],opset_version=17,dynamo=False)
import onnxruntime as ort
s=ort.InferenceSession(out,providers=["CPUExecutionProvider"])
for k in range(3):
    xx=torch.tensor(rng.normal(0,1,(104,896)),dtype=torch.float32); f2=torch.ones(1,100); f2[0,:[0,30,70][k]]=0
    with torch.no_grad(): a=float(w(xx,f2,qpool))
    b=float(s.run(None,{"h9":xx.numpy(),"fmask":f2.numpy(),"qpool":qpool.numpy()})[0][0])
    print("onnx vs torch (aleatório, máscara",[0,30,70][k],"):",a,b,abs(a-b))
print("tamanho MB",Path(out).stat().st_size/1e6)
# pesos JSON
def pack(d):
    return {k:{"shape":list(v.shape),"b64":base64.b64encode(np.ascontiguousarray(v,dtype=np.float32).tobytes()).decode()} for k,v in d.items()}
W={"vad":{"temperature":_VAD_TEMPERATURE,"tail_temperature":_VAD_TAIL_TEMPERATURE,"threshold":0.225,"qpool":pack({"qpool":qpool[0].numpy()})["qpool"],"prompt":prompt}}
for name,f in [("turn_end","turn_end/readout"),("same_speaker","diarization/same_speaker"),("speaker_change","diarization/speaker_change"),("speaker_count","diarization/speaker_count")]:
    W[name]=pack(load_np(str(S/(f+".safetensors"))))   # float64 de mean/scale viram float32 só no JSON
# turno: mean/scale ficam em float64 no original; guardar também como float64 para paridade
t=load_np(str(S/"turn_end/readout.safetensors"))
W["turn_end"]["mean64"]={"shape":[896],"b64":base64.b64encode(t["mean"].astype("<f8").tobytes()).decode()}
W["turn_end"]["scale64"]={"shape":[896],"b64":base64.b64encode(t["scale"].astype("<f8").tobytes()).decode()}
json.dump(W,open("../web/pesos.json","w"))
print("pesos.json MB",Path("../web/pesos.json").stat().st_size/1e6)
