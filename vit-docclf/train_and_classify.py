import argparse, csv, shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, get_cosine_schedule_with_warmup
from PIL import Image, ImageSequence
from pdf2image import convert_from_path
from sklearn.metrics import f1_score, accuracy_score

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif"}
PDF_EXT = ".pdf"

def pil_from_any(p: Path):
    if p.suffix.lower()==".pdf":
        return convert_from_path(str(p), dpi=200)[0].convert("RGB")
    im = Image.open(p)
    if getattr(im,"is_animated",False):
        im = next(ImageSequence.Iterator(im))
    return im.convert("RGB")

def tfm_train(s=224): 
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(int(s*1.2)),
        T.RandomResizedCrop(s,scale=(0.8,1.0)),
        T.RandomRotation(3),
        T.ColorJitter(0.1,0.1),
        T.ToTensor(),
        T.Normalize([0.5]*3,[0.5]*3)
    ])

def tfm_eval(s=224):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((s,s)),
        T.ToTensor(),
        T.Normalize([0.5]*3,[0.5]*3)
    ])

class DocDS(Dataset):
    def __init__(self,root,classes,train=True,size=224):
        self.root=Path(root); self.classes=classes
        self.c2i={c:i for i,c in enumerate(classes)}
        self.paths=[]; self.labels=[]
        t=tfm_train(size) if train else tfm_eval(size)
        self.t=t
        for c in classes:
            for f in (self.root/c).rglob("*"):
                if f.suffix.lower() in IMG_EXTS or f.suffix.lower()==PDF_EXT:
                    self.paths.append(f); self.labels.append(self.c2i[c])
    def __len__(self): return len(self.paths)
    def __getitem__(self,i): return self.t(pil_from_any(self.paths[i])), self.labels[i]

@torch.no_grad()
def evaluate(m,dl,dev):
    y_true=[];y_pred=[];m.eval()
    for x,y in dl:
        x=x.to(dev);y=torch.tensor(y).to(dev)
        pred=m(pixel_values=x).logits.argmax(-1)
        y_true+=y.tolist();y_pred+=pred.tolist()
    if not y_true: return 0,0
    return accuracy_score(y_true,y_pred), f1_score(y_true,y_pred,average='macro')

def train_vit(train_dir,val_dir,classes,ckpt="best.pt",img_size=224,batch=32,epochs=8,lr=3e-5):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tr=DocDS(train_dir,classes,True,img_size); va=DocDS(val_dir,classes,False,img_size)
    trl=DataLoader(tr,batch,shuffle=True,num_workers=2)
    val=DataLoader(va,batch*2,shuffle=False,num_workers=2)
    m=ViTForImageClassification.from_pretrained("google/vit-base-patch16-224",
        num_labels=len(classes),ignore_mismatched_sizes=True).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr)
    sched=get_cosine_schedule_with_warmup(opt,int(0.1*len(trl)*epochs),len(trl)*epochs)
    best=0
    for e in range(1,epochs+1):
        m.train(); tot=0
        for x,y in trl:
            x=x.to(dev);y=torch.tensor(y).to(dev)
            out=m(pixel_values=x,labels=y)
            out.loss.backward();opt.step();opt.zero_grad();sched.step()
            tot+=float(out.loss.item())
        acc,f1=evaluate(m,val,dev)
        print(f"[ep {e}] loss={tot/len(trl):.3f} acc={acc:.3f} f1={f1:.3f}")
        if f1>best:
            best=f1;torch.save({"state_dict":m.state_dict(),"classes":classes},ckpt)
            print(f"  saved {ckpt}")
    print("Training done ✅")

def load_model(ckpt):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load(ckpt,map_location=dev)
    m=ViTForImageClassification.from_pretrained("google/vit-base-patch16-224",
        num_labels=len(ck["classes"]),ignore_mismatched_sizes=True).to(dev)
    m.load_state_dict(ck["state_dict"]);m.eval()
    return m,ck["classes"],dev

@torch.no_grad()
def classify_folder(ckpt,input_dir,out_dir="sorted_output",img_size=224,move=False):
    m,classes,dev=load_model(ckpt);t=tfm_eval(img_size)
    inp=Path(input_dir);outp=Path(out_dir);outp.mkdir(exist_ok=True)
    for c in classes:(outp/c).mkdir(exist_ok=True)
    files=[p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXTS or p.suffix.lower()==PDF_EXT]
    with open(outp/"predictions.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["file","label","confidence"])
        for fp in files:
            x=t(pil_from_any(fp)).unsqueeze(0)
            logits=m(pixel_values=x.to(dev)).logits
            prob=torch.softmax(logits,-1);conf,pred=prob.max(-1)
            label=classes[pred.item()]
            w.writerow([str(fp),label,f"{conf.item():.4f}"])
            dest=(outp/label/fp.name)
            if move: shutil.move(str(fp),dest)
            else: shutil.copy2(str(fp),dest)
            print(f"{fp.name:30s} → {label:15s} ({conf.item():.2%})")
    print("Classification done ✅")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["train","classify","both"],required=True)
    ap.add_argument("--train_dir",default="data/train")
    ap.add_argument("--val_dir",default="data/val")
    ap.add_argument("--classes",nargs="+",default=["certificate","claim_form","invoice","policy","renewal"])
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--batch_size",type=int,default=32)
    ap.add_argument("--img_size",type=int,default=224)
    ap.add_argument("--ckpt",default="best.pt")
    ap.add_argument("--input_dir",default=None)
    ap.add_argument("--out_dir",default="sorted_output")
    ap.add_argument("--move",action="store_true")
    a=ap.parse_args()
    if a.mode in ("train","both"):
        train_vit(a.train_dir,a.val_dir,a.classes,a.ckpt,a.img_size,a.batch_size,a.epochs)
    if a.mode in ("classify","both"):
        classify_folder(a.ckpt,a.input_dir,a.out_dir,a.img_size,a.move)

if __name__=="__main__":
    main()
