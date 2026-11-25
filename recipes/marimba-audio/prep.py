import tarfile,csv,io,os,sys,random,json,soundfile as sf,urllib.request
from huggingface_hub import hf_hub_download
tsv=hf_hub_download("google/fleurs","data/pt_br/dev.tsv",repo_type="dataset")
tgz=hf_hub_download("google/fleurs","data/pt_br/audio/dev.tar.gz",repo_type="dataset")
rows={}
for r in csv.reader(open(tsv,encoding="utf-8"),delimiter="\t"):
    rows[r[1]]=dict(id=r[0],fn=r[1],raw=r[2],tr=r[3],n=int(r[5]) if r[5].isdigit() else 0)
os.makedirs("wav",exist_ok=True); keep=[]
with tarfile.open(tgz) as t:
    for m in t:
        fn=os.path.basename(m.name)
        if not fn.endswith(".wav") or fn not in rows: continue
        b=t.extractfile(m).read()
        try: d,sr=sf.read(io.BytesIO(b))
        except Exception: continue
        dur=len(d)/sr
        if dur<=8.0 and dur>=2.0:
            open(f"wav/{fn}","wb").write(b); keep.append(dict(rows[fn],dur=round(dur,2)))
json.dump(keep,open("clips.json","w"),ensure_ascii=False)
print("clips ≤8 s:",len(keep))
