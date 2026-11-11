"""Computer Vision AI-4002 - Lab 08: Fully reproducible Faster R-CNN detection.

Dataset: original generated RGB road-sign-like objects (circle, square, triangle).
Uses no external API, no BCCD samples, no pretrained model downloads.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign, box_iou

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'data'
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
CLASSES = ('__background__', 'circle', 'square', 'triangle')
PALETTE = {'circle': '#d63447', 'square': '#2878d0', 'triangle': '#d3a228'}
IMAGE_SIZE = 128


def fix_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def generate_one(rng, image_size=IMAGE_SIZE):
    """Make one synthetic detection image with non-overlapping annotated signs."""
    s = image_size
    base = int(rng.integers(203, 236))
    image_np = np.full((s,s,3),base,dtype=np.float32)
    yy,xx = np.mgrid[0:s,0:s]
    image_np += (xx/s-0.5)[...,None]*rng.uniform(-17,17)
    image_np += (yy/s-0.5)[...,None]*rng.uniform(-17,17)
    image_np += rng.normal(0,3,(s,s,1))
    image_np = np.uint8(np.clip(image_np,0,255))
    img = Image.fromarray(image_np,'RGB')
    d = ImageDraw.Draw(img)
    # A few faint straight lines act as unrelated background variation.
    for _ in range(int(rng.integers(0,4))):
        x1,y1,x2,y2=map(int,rng.integers(0,s,4))
        gray=int(rng.integers(181,213))
        d.line((x1,y1,x2,y2),fill=(gray,gray,gray),width=1)
    count = int(rng.integers(1,4))
    selected = []
    for _ in range(count):
        for attempt in range(100):
            size = int(rng.integers(23,40))
            x1 = int(rng.integers(5,s-size-5))
            y1 = int(rng.integers(5,s-size-5))
            x2,y2=x1+size,y1+size
            if all(x2+5<ax1 or ax2+5<x1 or y2+5<ay1 or ay2+5<y1 for _,ax1,ay1,ax2,ay2 in selected):
                break
        else:
            break
        label = str(rng.choice(CLASSES[1:]))
        if label=='circle':
            d.ellipse((x1,y1,x2,y2),fill=(221,52,68),outline=(90,28,28),width=2)
            c=(x1+x2)//2
            d.line((x1+size//4,(y1+y2)//2,x2-size//4,(y1+y2)//2),fill='white',width=max(3,size//7))
        elif label=='square':
            d.rounded_rectangle((x1,y1,x2,y2),radius=3,fill=(38,111,220),outline=(18,51,110),width=2)
            pad=size//4
            d.rectangle((x1+pad,y1+pad,x2-pad,y2-pad),outline=(245,245,252),width=3)
        else:
            d.polygon([(x1+size//2,y1),(x2,y2),(x1,y2)],fill=(239,193,39))
            d.line([(x1+size//2,y1),(x2,y2),(x1,y2),(x1+size//2,y1)],fill=(93,74,23),width=2)
            d.line((x1+size//2,y1+size//3,x1+size//2,y2-size//4),fill=(35,34,28),width=3)
            d.ellipse((x1+size//2-1,y2-size//6,x1+size//2+1,y2-size//6+2),fill=(35,34,28))
        selected.append((label,x1,y1,x2,y2))
    return img, selected


def generate(args):
    fix_seeds(args.seed)
    rng=np.random.default_rng(args.seed)
    DATA_ROOT.mkdir(exist_ok=True)
    counts={'train':args.n_train,'val':args.n_val,'test':args.n_test}
    manifest={'dataset':'Original Synthetic Road Sign Geometry Detection','seed':args.seed,
              'image_size':IMAGE_SIZE,'classes':list(CLASSES[1:]),'splits':counts,
              'annotation_format':'filename,class,xmin,ymin,xmax,ymax','coordinates':'xyxy in pixels'}
    for split,num in counts.items():
        split_dir=DATA_ROOT/split
        split_dir.mkdir(parents=True,exist_ok=True)
        rows=[]
        for i in range(num):
            img,objs=generate_one(rng)
            name=f'{split}_{i:04d}.png'
            img.save(split_dir/name)
            rows.extend([{'filename':name,'class':c,'xmin':x1,'ymin':y1,'xmax':x2,'ymax':y2}
                        for c,x1,y1,x2,y2 in objs])
        with (DATA_ROOT/f'{split}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['filename','class','xmin','ymin','xmax','ymax'])
            writer.writeheader();writer.writerows(rows)
        print(f'{split}: {num} images, {len(rows)} labeled objects, distribution {dict(Counter(r["class"] for r in rows))}',flush=True)
    (DATA_ROOT/'dataset_info.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    explore()


class ObjectDataset(Dataset):
    def __init__(self, split, augment=False):
        self.split=split
        self.augment=augment
        path=DATA_ROOT/f'{split}.csv'
        self.items=defaultdict(list)
        with path.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                self.items[r['filename']].append(r)
        self.filenames=sorted(self.items)
        if not self.filenames:
            raise RuntimeError(f'No images found for split={split}. Run generate first.')
        self.to_tensor=transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,i):
        name=self.filenames[i]
        im=Image.open(DATA_ROOT/self.split/name).convert('RGB')
        rows=self.items[name]
        boxes=torch.as_tensor([[float(r[k]) for k in ('xmin','ymin','xmax','ymax')] for r in rows],dtype=torch.float32)
        labels=torch.as_tensor([CLASSES.index(r['class']) for r in rows],dtype=torch.int64)
        if self.augment and random.random()<0.5:
            im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            old=boxes.clone()
            boxes[:,0]=im.width-old[:,2]
            boxes[:,2]=im.width-old[:,0]
        assert (boxes[:,2]>boxes[:,0]).all() and (boxes[:,3]>boxes[:,1]).all(), name
        area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        target={'boxes':boxes,'labels':labels,'image_id':torch.tensor(i),
                'area':area,'iscrowd':torch.zeros(len(boxes),dtype=torch.int64)}
        return self.to_tensor(im),target


def collate(batch):
    return tuple(zip(*batch))


class TinyBackbone(nn.Module):
    """Custom single-feature-map CNN backbone for actual Faster R-CNN."""
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,24,3,padding=1,stride=2),nn.BatchNorm2d(24),nn.ReLU(inplace=True),
            nn.Conv2d(24,48,3,padding=1,stride=2),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
            nn.Conv2d(48,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),nn.ReLU(inplace=True))
        self.out_channels=64

    def forward(self,x):
        return self.features(x)


def build_model():
    backbone=TinyBackbone()
    anchors=AnchorGenerator(sizes=((16,24,32,48,64),),aspect_ratios=((0.75,1.,1.33),))
    pool=MultiScaleRoIAlign(featmap_names=['0'],output_size=4,sampling_ratio=2)
    head=TwoMLPHead(64*4*4,128)
    predictor=FastRCNNPredictor(128,len(CLASSES))
    return FasterRCNN(backbone,num_classes=None,rpn_anchor_generator=anchors,
                      box_roi_pool=pool,box_head=head,box_predictor=predictor,
                      min_size=IMAGE_SIZE,max_size=IMAGE_SIZE,
                      rpn_pre_nms_top_n_train=160,rpn_post_nms_top_n_train=80,
                      rpn_pre_nms_top_n_test=130,rpn_post_nms_top_n_test=60,
                      rpn_batch_size_per_image=128,box_batch_size_per_image=128,
                      box_detections_per_img=25,box_score_thresh=0.01,
                      box_nms_thresh=0.40)


def explore():
    FIG.mkdir(exist_ok=True)
    train=ObjectDataset('train')
    counts=Counter(r['class'] for rows in train.items.values() for r in rows)
    fig,ax=plt.subplots(figsize=(7,3.4))
    lab=list(CLASSES[1:]); ys=[counts.get(c,0) for c in lab]
    ax.bar(lab,ys,color=[PALETTE[c] for c in lab])
    for i,v in enumerate(ys):ax.text(i,v+0.5,str(v),ha='center')
    ax.set_ylim(0,max(ys)*1.18 if ys else 1);ax.set_ylabel('Number of object instances');ax.set_title('Training class distribution')
    fig.tight_layout();fig.savefig(FIG/'class_distribution.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(10,7))
    for i,ax in enumerate(axs.flat):
        name=train.filenames[i]
        ax.imshow(Image.open(DATA_ROOT/'train'/name))
        for r in train.items[name]:
            x1,y1,x2,y2=[int(r[k]) for k in ('xmin','ymin','xmax','ymax')]
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,lw=2,ec=PALETTE[r['class']]))
            ax.text(x1,max(4,y1-3),r['class'],fontsize=8,color='black',bbox={'facecolor':'white','alpha':0.7,'pad':1})
        ax.axis('off')
    fig.suptitle('Ground truth: sample images and bounding boxes',fontsize=14)
    fig.tight_layout();fig.savefig(FIG/'ground_truth_examples.png',dpi=170);plt.close(fig)
    summary={'images':{s:len(ObjectDataset(s)) for s in ('train','val','test')},
             'train_instances_per_class':dict(counts)}
    OUT.mkdir(exist_ok=True)
    (OUT/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def compute_ap(rec,prec):
    """All-point integral AP interpolated over a precision envelope."""
    mrec=np.concatenate(([0.],np.asarray(rec,dtype=float),[1.]))
    mpre=np.concatenate(([0.],np.asarray(prec,dtype=float),[0.]))
    for i in range(len(mpre)-2,-1,-1):mpre[i]=max(mpre[i],mpre[i+1])
    indexes=np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indexes+1]-mrec[indexes])*mpre[indexes+1]))


@torch.no_grad()
def evaluate(model, loader, device, score_thresh=0.35, iou_thresh=0.5):
    """Report classwise AP@0.50 and score-thresholded precision/recall/F1."""
    model.eval()
    records={i:[] for i in range(1,len(CLASSES))}
    n_gt=Counter()
    totals={'tp':0,'fp':0,'fn':0}
    matched_ious=[]
    per_image=[]
    for images,targets in loader:
        preds=model([im.to(device) for im in images])
        for pred,target in zip(preds,targets):
            pboxes=pred['boxes'].cpu(); plabels=pred['labels'].cpu(); pscores=pred['scores'].cpu()
            gtboxes=target['boxes'].cpu(); gtlabels=target['labels'].cpu()
            n_gt.update(int(k) for k in gtlabels.tolist())
            image_stats={'tp':0,'fp':0,'fn':0}
            for label in range(1,len(CLASSES)):
                b=gtboxes[gtlabels==label]
                idx=(plabels==label).nonzero(as_tuple=True)[0]
                idx=idx[torch.argsort(pscores[idx],descending=True)]
                claimed=set(); matched_at_thresh=0; all_at_thresh=0
                for pi in idx.tolist():
                    confidence=float(pscores[pi])
                    best_iou=0.0; best_j=None
                    if len(b):
                        ious=box_iou(pboxes[pi].unsqueeze(0),b).squeeze(0)
                        for j in torch.argsort(ious,descending=True).tolist():
                            if j not in claimed:
                                best_iou=float(ious[j]);best_j=j;break
                    tp=best_j is not None and best_iou>=iou_thresh
                    if tp:claimed.add(best_j)
                    records[label].append((confidence,int(tp)))
                    if confidence>=score_thresh:
                        all_at_thresh+=1
                        if tp:
                            matched_at_thresh+=1
                            matched_ious.append(best_iou)
                # AP uses every detection; thresholded stats use only high-score matches.
                # Score-ordered matching above may let subthreshold boxes claim a GT;
                # since scores descend, threshold-qualified detections are unaffected.
                image_stats['tp']+=matched_at_thresh
                image_stats['fp']+=all_at_thresh-matched_at_thresh
                image_stats['fn']+=max(0,len(b)-matched_at_thresh)
            for k in totals:totals[k]+=image_stats[k]
            per_image.append(image_stats)
    aps={}
    for i in range(1,len(CLASSES)):
        recs=sorted(records[i],key=lambda x:x[0],reverse=True)
        if n_gt[i]==0:
            aps[CLASSES[i]]=None;continue
        tp=np.cumsum([v[1] for v in recs]);fp=np.cumsum([1-v[1] for v in recs])
        recall=tp/n_gt[i];precision=tp/np.maximum(tp+fp,1)
        aps[CLASSES[i]]=round(compute_ap(recall,precision),4)
    p=totals['tp']/max(totals['tp']+totals['fp'],1)
    r=totals['tp']/max(totals['tp']+totals['fn'],1)
    f1=2*p*r/(p+r) if p+r else 0
    return {'AP50_per_class':aps,'mAP50':round(float(np.mean([x for x in aps.values() if x is not None])),4),
            'precision_at_threshold':round(p,4),'recall_at_threshold':round(r,4),
            'f1_at_threshold':round(f1,4),'threshold':score_thresh,'iou_threshold':iou_thresh,
            'true_positive':totals['tp'],'false_positive':totals['fp'],'false_negative':totals['fn'],
            'ground_truth_instances':sum(n_gt.values()),'evaluated_images':len(per_image),
            'mean_matched_iou':round(float(np.mean(matched_ious)),4) if matched_ious else None}


def train(args):
    fix_seeds(args.seed)
    OUT.mkdir(exist_ok=True)
    device=torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_ds=ObjectDataset('train',augment=True);val_ds=ObjectDataset('val')
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate)
    val_loader=DataLoader(val_ds,batch_size=2,shuffle=False,num_workers=0,collate_fn=collate)
    model=build_model().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    history=[];best=-1.;best_epoch=0;overall_start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();epoch_loss=0.;start=time.perf_counter()
        for step,(images,targets) in enumerate(train_loader,1):
            images=[a.to(device) for a in images]
            targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
            loss_dict=model(images,targets)
            loss=sum(loss_dict.values())
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss: {loss_dict}')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step();epoch_loss+=float(loss.detach())
        result=evaluate(model,val_loader,device)
        row={'epoch':epoch,'train_loss':round(epoch_loss/len(train_loader),5),
             'val_mAP50':result['mAP50'],'elapsed_seconds':round(time.perf_counter()-start,1)}
        history.append(row)
        print(f'Epoch {epoch}/{args.epochs}: loss={row["train_loss"]:.4f}, val mAP@.50={row["val_mAP50"]:.3f}, time={row["elapsed_seconds"]:.1f}s',flush=True)
        if result['mAP50']>=best:
            best=result['mAP50'];best_epoch=epoch
            torch.save({'model_state':model.state_dict(),'epoch':epoch,'class_names':list(CLASSES),
                        'dataset':'Synthetic signs (non-BCCD)','seed':args.seed,
                        'val_metrics':result}, OUT/'best_model.pth')
    (OUT/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    with (OUT/'training_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    plot_history(history)
    report={'best_epoch':best_epoch,
            'best_val_mAP50':best,'training_seconds':round(time.perf_counter()-overall_start,1),
            'device':str(device),'epochs':args.epochs,'batch_size':args.batch_size,'learning_rate':args.lr}
    (OUT/'training_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def plot_history(history):
    FIG.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    epochs=[x['epoch'] for x in history]
    axs[0].plot(epochs,[x['train_loss'] for x in history],'o-');axs[0].set_title('Training loss');axs[0].set_xlabel('Epoch')
    axs[1].plot(epochs,[x['val_mAP50'] for x in history],'o-');axs[1].set_title('Validation mAP@0.50');axs[1].set_xlabel('Epoch');axs[1].set_ylim(0,1)
    for ax in axs:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(FIG/'training_curves.png',dpi=170);plt.close(fig)


def load_trained(device):
    path=OUT/'best_model.pth'
    if not path.exists():raise FileNotFoundError(f'No model at {path}. Run train first.')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=build_model();model.load_state_dict(checkpoint['model_state']);model.to(device).eval()
    return model,checkpoint


def test(args):
    device=torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model,checkpoint=load_trained(device)
    ds=ObjectDataset('test')
    result=evaluate(model,DataLoader(ds,batch_size=2,collate_fn=collate,num_workers=0),device,score_thresh=args.threshold)
    ap75=evaluate(model,DataLoader(ds,batch_size=2,collate_fn=collate,num_workers=0),device,score_thresh=args.threshold,iou_thresh=.75)
    result['mAP75']=ap75['mAP50']
    result['AP75_per_class']=ap75['AP50_per_class']
    result['trained_checkpoint_epoch']=checkpoint['epoch']
    (OUT/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('HELD-OUT TEST RESULTS:',json.dumps(result,indent=2),flush=True)
    visualize_predictions(model,ds,device,threshold=args.threshold,max_images=args.preview_images)
    export_test_detections(model,ds,device,threshold=args.threshold)
    return result


@torch.no_grad()
def visualize_predictions(model,ds,device,threshold=0.35,max_images=6):
    FIG.mkdir(exist_ok=True)
    n=min(max_images,len(ds)); cols=2;rows=math.ceil(n/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(10,rows*4.7),squeeze=False)
    for i,ax in enumerate(axes.flat):
        if i>=n:ax.axis('off');continue
        im,tgt=ds[i]
        pred=model([im.to(device)])[0]
        ax.imshow(im.permute(1,2,0).numpy())
        for b,l in zip(tgt['boxes'],tgt['labels']):
            x1,y1,x2,y2=map(float,b)
            color=PALETTE[CLASSES[int(l)]]
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec=color,lw=2,ls=':'))
        for b,l,score in zip(pred['boxes'],pred['labels'],pred['scores']):
            if float(score)<threshold:continue
            x1,y1,x2,y2=map(float,b)
            color=PALETTE[CLASSES[int(l)]]
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec=color,lw=2))
            ax.text(x1,min(127,y1+2),f'{CLASSES[int(l)]} {float(score):.2f}',fontsize=7,color='black',
                    bbox={'facecolor':'white','edgecolor':'none','alpha':0.7,'pad':1})
        ax.set_title(ds.filenames[i],fontsize=10)
        ax.axis('off')
    fig.suptitle('Test predictions: solid = detected | dotted = ground truth',fontsize=12)
    fig.tight_layout();fig.savefig(FIG/'test_predictions.png',dpi=170);plt.close(fig)
    print('Saved visual predictions to figures/test_predictions.png',flush=True)



@torch.no_grad()
def export_test_detections(model,ds,device,threshold=0.35):
    """Save predicted boxes to CSV so every detection can be inspected."""
    rows=[]
    for i in range(len(ds)):
        image,_=ds[i]
        pred=model([image.to(device)])[0]
        for box,label,score in zip(pred['boxes'],pred['labels'],pred['scores']):
            if float(score)<threshold:continue
            b=list(map(float,box))
            rows.append(dict(filename=ds.filenames[i],class_name=CLASSES[int(label)],
                             score=round(float(score),5),xmin=round(b[0],2),ymin=round(b[1],2),
                             xmax=round(b[2],2),ymax=round(b[3],2)))
    with (OUT/'test_detections.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['filename','class_name','score','xmin','ymin','xmax','ymax'])
        writer.writeheader();writer.writerows(rows)
    print(f'Saved {len(rows)} test detections to outputs/test_detections.csv')

@torch.no_grad()
def predict_user_image(args):
    """Apply trained detector to any user-supplied image and save visualization + JSON."""
    if not args.image:raise ValueError('Provide --image /path/to/image.png')
    image_path=Path(args.image)
    if not image_path.is_file():raise FileNotFoundError(image_path)
    device=torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model,_=load_trained(device)
    img=Image.open(image_path).convert('RGB')
    arr=transforms.ToTensor()(img).to(device)
    out=model([arr])[0]
    fig,ax=plt.subplots(figsize=(7,7))
    ax.imshow(img)
    predictions=[]
    for box,label,score in zip(out['boxes'],out['labels'],out['scores']):
        if float(score)<args.threshold:continue
        x1,y1,x2,y2=list(map(float,box))
        name=CLASSES[int(label)]
        predictions.append({'class':name,'confidence':round(float(score),4),'bbox_xyxy':[round(x,2) for x in (x1,y1,x2,y2)]})
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,ec=PALETTE[name],fill=False,lw=2))
        ax.text(x1,y1,f'{name} {float(score):.2f}',fontsize=10,bbox=dict(facecolor='white',alpha=.8,pad=2))
    ax.axis('off');fig.tight_layout()
    outpath=OUT/(image_path.stem+'_prediction.png')
    fig.savefig(outpath,dpi=160);plt.close(fig)
    jsonpath=OUT/(image_path.stem+'_prediction.json')
    jsonpath.write_text(json.dumps({'input':str(image_path),'predictions':predictions},indent=2),encoding='utf-8')
    print(f'Saved {len(predictions)} detections to {outpath} and {jsonpath}')


def smoke_test():
    fix_seeds(42)
    ds=ObjectDataset('train');im,tgt=ds[0]
    assert im.shape==(3,IMAGE_SIZE,IMAGE_SIZE)
    assert tgt['boxes'].shape[1]==4
    assert (tgt['labels']>0).all()
    assert abs(compute_ap([.5,1.],[1.,1.])-1.0)<1e-6
    model=build_model()
    with torch.no_grad():
        model.eval();out=model([im])[0]
    assert set(('boxes','scores','labels')).issubset(out)
    print(f'SMOKE TEST PASSED: valid annotations, AP integration, model inference; {len(out["boxes"])} raw detections')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('generate','explore','train','test','predict','all','smoke'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--n-train',type=int,default=180)
    p.add_argument('--n-val',type=int,default=36)
    p.add_argument('--n-test',type=int,default=36)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threshold',type=float,default=.35)
    p.add_argument('--preview-images',type=int,default=6)
    p.add_argument('--device',choices=('cpu','cuda'),default=None)
    p.add_argument('--image',type=str,default=None,help='Image path for predict command')
    return p.parse_args()


if __name__=='__main__':
    a=parse_args()
    if a.command in ('generate','all'):generate(a)
    if a.command=='explore':print(json.dumps(explore(),indent=2))
    if a.command=='smoke':smoke_test()
    if a.command in ('train','all'):train(a)
    if a.command in ('test','all'):test(a)
    if a.command=='predict':predict_user_image(a)
