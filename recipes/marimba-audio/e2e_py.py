"""Paridade de ponta a ponta em Python: mel do torch -> torre ONNX -> cabeça ONNX + leituras em numpy (mesma lógica do JS) vs decisões do torch."""
import json,numpy as np,onnxruntime as ort,time
from marimba.readout import pool_layer,predict_readout
from marimba._diarization import task_features,DiarizationReadout
from safetensors.numpy import load_file
from pathlib import Path
S=sorted(Path.home().glob(".cache/huggingface/hub/models--charcot-ai--marimba/snapshots/*"))[0]
ref=json.load(open("ref/ref.json")); W=json.load(open("../web/pesos.json"))
import base64; qpool=np.frombuffer(base64.b64decode(W["vad"]["qpool"]["b64"]),dtype=np.float32)[None]
tower=ort.InferenceSession("audio_tower.onnx",providers=["CPUExecutionProvider"]); head=ort.InferenceSession("vad_head.onnx",providers=["CPUExecutionProvider"])
M=128*800; melA=np.fromfile("ref/melA.f32",dtype=np.float32).reshape(56,128,800); melB=np.fromfile("ref/melB.f32",dtype=np.float32).reshape(56,128,800)
turn_w=load_file(str(S/"turn_end/readout.safetensors")); dz={t:DiarizationReadout(t,load_file(str(S/f"diarization/{t}.safetensors"))) for t in ("same_speaker","speaker_change","speaker_count")}
f16=lambda a:a.astype(np.float16).astype(np.float32)
sig=lambda x:1/(1+np.exp(-x)); T=W["vad"]["temperature"]
rA=[];rB=[];tt=[]
for i in range(56):
    n=ref["A"][i]["n"]; mask=np.zeros(128000,bool); mask[-n:]=True
    t=time.time(); h9,h12,h18,af=tower.run(None,{"mel":melA[i]}); tt.append(time.time()-t)
    fm=mask.reshape(100,-1).any(1).astype(np.float32)[None]
    sc=float(head.run(None,{"h9":h9,"fmask":fm,"qpool":qpool})[0][0]); ts,tp=predict_readout(turn_w,pool_layer(f16(h18),mask))
    rA.append((abs(sig(sc/T)-ref["A"][i]["vad_p"]),abs(tp-ref["A"][i]["turn_p"]),abs(sc-ref["A"][i]["vad_score"])))
    h9,h12,h18,af=tower.run(None,{"mel":melB[i]}); f=f16(h12)
    r=ref["B"][i]; same=float(dz["same_speaker"].predict(task_features("same_speaker",f)[None])[0][1]); ch=float(dz["speaker_change"].predict(task_features("speaker_change",f)[None])[0][1]); co=dz["speaker_count"].predict(task_features("speaker_count",f)[None])[0]
    rB.append((abs(same-r["same_p"]),abs(ch-r["change_p"]),float(np.abs(co-np.array(r["count_p"])).max())))
a=np.array(rA);b=np.array(rB)
res=dict(vad_p_maxdiff=a[:,0].max(),turn_p_maxdiff=a[:,1].max(),vad_score_maxdiff=a[:,2].max(),same_maxdiff=b[:,0].max(),change_maxdiff=b[:,1].max(),count_maxdiff=b[:,2].max(),tower_cpu_s_median=float(np.median(tt)))
print(json.dumps(res,indent=1)); json.dump(res,open("../paridade_py_e2e.json","w"),indent=1)
