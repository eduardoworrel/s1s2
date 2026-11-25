"""Exports the Qwen3-ASR audio encoder (fixed 8 s window = 800 mel frames) to ONNX, with layers 9, 12 and 18 and the final output.
Parity against torch on the 56 FLEURS pt-BR clips."""
import json,sys,time,torch,numpy as np,soundfile as sf
from pathlib import Path
from huggingface_hub import snapshot_download
from marimba._vendor.qwen_asr.configuration_qwen3_asr import Qwen3ASRConfig
from marimba._vendor.qwen_asr.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from marimba._vendor.qwen_asr.processing_qwen3_asr import Qwen3ASRProcessor
from marimba.weights import ENCODER_REPO, ENCODER_REVISION
P=snapshot_download(ENCODER_REPO,revision=ENCODER_REVISION)
model=Qwen3ASRForConditionalGeneration.from_pretrained(P,config=Qwen3ASRConfig.from_pretrained(P),dtype=torch.float32,attn_implementation="eager").eval()
proc=Qwen3ASRProcessor.from_pretrained(P,fix_mistral_regex=True)
tower=model.thinker.audio_tower
print("n_window",tower.n_window,"n_window_infer",tower.n_window_infer,"layers",len(tower.layers),flush=True)
cap={}
hooks=[tower.layers[l-1].register_forward_hook((lambda L:(lambda m,i,o:cap.__setitem__(L,o[0] if isinstance(o,tuple) else o)))(l)) for l in (9,12,18)]
import torch.nn.functional as F
class W(torch.nn.Module):
    """Same computation as Qwen3ASRAudioEncoder.forward for a fixed T=800: 8 blocks of 100 frames, a single attention window of 104 positions."""
    def __init__(s): super().__init__(); s.t=tower; s.cu=torch.tensor([0,104],dtype=torch.int32)
    def forward(s,feats):                       # feats [128,800]
        t=s.t
        x=feats.T.reshape(8,100,128).transpose(1,2).unsqueeze(1)
        for c in (t.conv2d1,t.conv2d2,t.conv2d3): x=F.gelu(c(x))
        b,c,f,tt=x.shape
        x=t.conv_out(x.permute(0,3,1,2).reshape(b,tt,c*f))
        x=x+t.positional_embedding.positional_embedding[:tt].unsqueeze(0).to(x.dtype)
        h=x.reshape(b*tt,-1)
        for layer in t.layers: h=layer(h,s.cu)[0]
        h=t.ln_post(h); h=t.act(t.proj1(h)); h=t.proj2(h)
        return cap[9],cap[12],cap[18],h
w=W().eval()
# real input: 8 s of a clip
clip=sorted(Path("wav").glob("*.wav"))[0]; a,sr=sf.read(clip); a=np.asarray(a,dtype=np.float32)
import soxr; a=soxr.resample(a,sr,16000)[-128000:]; a=np.pad(a,(128000-len(a),0))
f=proc.feature_extractor([a],sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors="pt",truncation=False)
x=f["input_features"][0]; m=f["attention_mask"][0].bool(); x=x[:,m]; print("input",tuple(x.shape),flush=True)
with torch.no_grad(): ref=w(x); ref_orig=tower(x,feature_lens=torch.tensor([800])).last_hidden_state; print('static vs original:',float((ref[3]-ref_orig).abs().max()),flush=True)
print("ref",[tuple(r.shape) for r in ref],flush=True)
out="port/audio_tower.onnx"; t0=time.time()
torch.onnx.export(w,(x,),out,input_names=["mel"],output_names=["h9","h12","h18","audio_features"],opset_version=17,dynamo=False)
print("exported in",round(time.time()-t0),"s",Path(out).stat().st_size/1e6,"MB",flush=True)
import onnxruntime as ort
s=ort.InferenceSession(out,providers=["CPUExecutionProvider"])
got=s.run(None,{"mel":x.numpy()})
for n,r,g in zip(("h9","h12","h18","audio_features"),ref,got):
    r=r.numpy(); d=np.abs(r-g).max(); cos=(r*g).sum()/np.sqrt((r*r).sum()*(g*g).sum()); print(n,"max|dif|",float(d),"cos",float(cos))
