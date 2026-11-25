"""Torch references: mel, windows and decisions (voice activity, turn end, speaker) on the 56 clips.
Set A: the clip as an 8 s window, right-aligned (zero-padded). Set B: full 8 s windows made by concatenating consecutive clips (for the speaker readers, which only run on a full window)."""
import json,numpy as np,time
from marimba import Marimba
from marimba.audio import decode_audio,right_aligned_window
from marimba.pretrained import _render_vad_prompt,_CANONICAL_VAD_STEM,_VAD_TEMPERATURE,sigmoid
clips=json.load(open("../clips.json"))
m=Marimba.from_pretrained(backend="torch",device="cpu",dtype="float32"); eng=m._engine
auds=[decode_audio(f"../wav/{c['fn']}") for c in clips]
def run(win,mask,diar):
    enc=eng.encode(win); f=enc["features"]
    r={}
    sc=eng.head_score(f,mask,_render_vad_prompt(_CANONICAL_VAD_STEM)); r["vad_score"]=sc; r["vad_p"]=sigmoid(sc/_VAD_TEMPERATURE)
    ts,tp=eng.turn_end(f,mask); r["turn_score"]=float(ts); r["turn_p"]=float(tp)
    if diar:
        d=eng.diarization(f,{"same_speaker","speaker_change","speaker_count"})
        r["same_p"]=float(d["same_speaker"][1]); r["change_p"]=float(d["speaker_change"][1]); r["count_p"]=[float(x) for x in d["speaker_count"]]
    return r,f
A=[];B=[];melA=[];melB=[];wA=[];wB=[]
def mel(win):
    x=eng.processor.feature_extractor([win],sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors="pt",truncation=False)
    mm=x["attention_mask"][0].bool(); return x["input_features"][0][:,mm].numpy()
t0=time.time()
for i,c in enumerate(clips):
    win,mask=right_aligned_window(auds[i]); r,_=run(win,mask,False); r["fn"]=c["fn"]; r["n"]=int(mask.sum()); A.append(r); melA.append(mel(win)); wA.append(win)
    cat=np.concatenate([auds[i],auds[(i+1)%len(clips)],auds[(i+2)%len(clips)]]); win2,mask2=right_aligned_window(cat)
    assert mask2.all()
    r2,_=run(win2,mask2,True); r2["fn"]=c["fn"]; B.append(r2); melB.append(mel(win2)); wB.append(win2)
    print(i,round(time.time()-t0),r["vad_p"],r["turn_p"],r2["same_p"],r2["change_p"],r2["count_p"],flush=True)
json.dump({"A":A,"B":B},open("ref/ref.json","w"))
for n,a in (("melA",melA),("melB",melB),("winA",wA),("winB",wB)): np.asarray(a,dtype=np.float32).tofile(f"ref/{n}.f32")
print("mel shape",melA[0].shape)
