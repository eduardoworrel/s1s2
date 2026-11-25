import json,random,re,time,unicodedata,sys
from marimba import Marimba, Question
random.seed(11)
norm=lambda s:"".join(c for c in unicodedata.normalize("NFKD",s.lower()) if not unicodedata.combining(c))
clips=json.load(open("clips.json"))
toks=lambda s:re.findall(r"[a-zà-ú]+",norm(s))
words={c["fn"]:set(toks(c["tr"])) for c in clips}
allw=sorted({w for s in words.values() for w in s if len(w)>=5})
t0=time.time(); model=Marimba.from_pretrained(backend="torch",device="cpu",dtype="float32"); print("load s:",round(time.time()-t0,1),flush=True)
rows=[];lid=[]
for i,c in enumerate(clips):
    cand=[w for w in words[c["fn"]] if len(w)>=5]
    if not cand: continue
    pos=random.choice(cand); neg=random.choice([w for w in allw if w not in words[c["fn"]]])
    qs=[Question.noul(f'Did someone say "{pos}"?'),Question.noul(f'Did someone say "{neg}"?'),Question.choice("What language?",["en","pt","es"]),Question.noul("Is someone speaking?")]
    t1=time.time(); ds=model(f"wav/{c['fn']}",qs); dt=time.time()-t1
    d=[x.to_dict() for x in ds]
    rows.append(dict(fn=c["fn"],pos=pos,neg=neg,p_pos=d[0].get("probability"),p_neg=d[1].get("probability"),lid=d[2].get("value"),vad=d[3].get("probability"),s=round(dt,2),dur=c["dur"]))
    if i%10==0: print(i,len(clips),rows[-1],flush=True)
json.dump(rows,open("result.json","w"),ensure_ascii=False)
