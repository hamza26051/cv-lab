"""Rebuild the lab's submission-ready PDF report from recorded experimental results.
Requires reportlab. Run: python make_report.py
"""
from pathlib import Path
from collections import Counter
import csv
import json
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
                                KeepTogether, HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'Lab08_Faster_RCNN_Report.pdf'
metrics=json.loads((ROOT/'outputs'/'test_metrics.json').read_text())
summary=json.loads((ROOT/'outputs'/'training_summary.json').read_text())
data=json.loads((ROOT/'data'/'dataset_info.json').read_text())
with (ROOT/'data'/'train.csv').open(newline='') as f:
    class_counts=Counter(r['class'] for r in csv.DictReader(f))

REG='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
if Path(REG).exists() and Path(BOLD).exists():
    pdfmetrics.registerFont(TTFont('ReportSans',REG))
    pdfmetrics.registerFont(TTFont('ReportSans-Bold',BOLD))
    face='ReportSans';bold_face='ReportSans-Bold'
else:
    face='Helvetica';bold_face='Helvetica-Bold'
navy=colors.HexColor('#142E45');teal=colors.HexColor('#087E8B');subtle=colors.HexColor('#516778')
light=colors.HexColor('#ECF4F6');dark=colors.HexColor('#263B4A')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle',fontName=bold_face,fontSize=21,leading=29,textColor=navy,spaceAfter=10))
styles.add(ParagraphStyle(name='ReportSubtitle',fontName=face,fontSize=10.5,leading=16,textColor=subtle,spaceAfter=14))
styles.add(ParagraphStyle(name='SectionTitle',fontName=bold_face,fontSize=12.7,leading=17,textColor=navy,spaceBefore=12,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='SubsectionTitle',fontName=bold_face,fontSize=10,leading=14,textColor=teal,spaceBefore=9,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle(name='BodyReport',fontName=face,fontSize=8.75,leading=13.1,textColor=dark,spaceAfter=7))
styles.add(ParagraphStyle(name='SmallReport',fontName=face,fontSize=7.6,leading=11,textColor=dark,spaceAfter=5))
styles.add(ParagraphStyle(name='CaptionReport',fontName=face,fontSize=7.5,leading=11,textColor=subtle,alignment=TA_CENTER,spaceAfter=11))
styles.add(ParagraphStyle(name='TableHeadReport',fontName=bold_face,fontSize=7.9,leading=11,textColor=navy))
styles.add(ParagraphStyle(name='TableCellReport',fontName=face,fontSize=7.9,leading=11,textColor=dark))

def P(text,sty='BodyReport'):return Paragraph(text,styles[sty])
def H(text):story.append(P(text,'SectionTitle'))
def SH(text):story.append(P(text,'SubsectionTitle'))
def para(text):story.append(P(text))
def caption(text):story.append(P(text,'CaptionReport'))
def add_table(rows,widths=None):
    tbl=Table([[P(str(x),'TableHeadReport' if i==0 else 'TableCellReport') for x in row] for i,row in enumerate(rows)],colWidths=widths,repeatRows=1,hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),light),('BOX',(0,0),(-1,-1),.35,colors.HexColor('#D0DBE1')),
        ('LINEBELOW',(0,0),(-1,0),.8,teal),('LINEBELOW',(0,1),(-1,-2),.25,colors.HexColor('#DFE7EA')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    story.append(tbl);story.append(Spacer(1,10))
def fig(name,width,maxheight=None):
    from PIL import Image as PILImage
    path=ROOT/'figures'/name
    w,h=PILImage.open(path).size
    hh=width*h/w
    if maxheight and hh>maxheight:
        hh=maxheight;width=maxheight*w/h
    story.append(Image(str(path),width=width,height=hh,hAlign='CENTER'))

def page(canvas,doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(teal);canvas.setLineWidth(1.2)
    canvas.line(39,h-37,w-39,h-37)
    canvas.setFont(bold_face,7.3);canvas.setFillColor(navy)
    canvas.drawString(40,h-28,'COMPUTER VISION  |  AI-4002  |  LAB 08')
    canvas.setStrokeColor(colors.HexColor('#D8E4E8'))
    canvas.line(39,43,w-39,43)
    canvas.setFont(face,7.1);canvas.setFillColor(subtle)
    canvas.drawString(40,29,'Faster R-CNN  /  Synthetic geometric-object dataset  /  Reproducible experiment')
    canvas.drawRightString(w-40,29,f'Page {doc.page}')
    canvas.restoreState()

story=[]
story.append(Spacer(1,20))
story.append(P('LAB 08', 'ReportSubtitle'))
story.append(P('Object Detection Using<br/>Faster R-CNN', 'ReportTitle'))
story.append(P('Complete implementation, training, and held-out evaluation on a non-BCCD dataset', 'ReportSubtitle'))
story.append(HRFlowable(width='100%',thickness=2,color=teal,spaceBefore=2,spaceAfter=15))
add_table([['Course','Student details','Submission'],['Computer Vision (AI-4002)','Name: ____________________<br/>Roll no: ____________________','Lab Task 6.1<br/>Lab 08 Manual']],widths=[168,185,155])
H('1. Abstract and objective')
para('This lab implements an end-to-end region-based object detector using <b>Faster R-CNN</b>. It follows the final requirement on page 17 of the provided Lab 08 manual: select a dataset other than Blood Cell Count and Detection (BCCD), then identify and localize objects with bounding boxes. The chosen dataset is an <b>original, programmatically generated collection of colored geometric sign-like objects</b>, not a public real-world dataset. Classes are circle, square and triangle.')
para('The completed workflow generates labeled images, inspects class distributions and ground-truth boxes, creates a custom PyTorch Dataset, trains a Faster R-CNN model, validates it during training, evaluates a held-out test split, and saves both annotated images and prediction data.')
H('2. Experiment at a glance')
add_table([['Property','Recorded value'],['Train / validation / test images','150 / 30 / 30 (210 total)'],
           ['Training labeled instances',str(sum(class_counts.values()))],
           ['Test labeled instances',str(metrics['ground_truth_instances'])],
           ['Backbone','Lightweight 4-convolution CNN (learned from scratch)'],
           ['Trainable detector','Faster R-CNN: RPN + RoI Align + classification / box regression'],
           ['Optimizer','Adam, learning rate 0.001, batch size 4, 12 epochs'],
           ['Model selection','Highest validation mAP@0.50; checkpoint epoch '+str(metrics['trained_checkpoint_epoch'])]],widths=[160,348])
H('3. R-CNN family and algorithm')
para('The manual distinguishes three region-based methods. Original <b>R-CNN</b> processes separately selected image regions, making it computationally expensive. <b>Fast R-CNN</b> shares feature extraction for a whole image but still relies on an external proposal mechanism. <b>Faster R-CNN</b> replaces external selective search with a learned Region Proposal Network (RPN), sharing image features between region generation and recognition. This implementation uses Faster R-CNN, which is explicitly covered in sections 2 and 5 of the manual.')
para('<b>Pipeline:</b> RGB image (128 x 128) -&gt; shared CNN features -&gt; RPN candidate regions -&gt; RoI Align (fixed-size region tensors) -&gt; small fully connected detection head -&gt; class scores and refined (x1, y1, x2, y2) boxes.')
H('4. Dataset preparation and exploration')
para('The dataset is procedurally produced with random positions, object sizes, minor background noise and faint lines. Images contain one to three objects. No BCCD image or annotation is used. The seed is 42; the generated train, validation and test examples are distinct. Annotations are stored in three CSV files with columns: <b>filename, class, xmin, ymin, xmax, ymax</b>. The loader converts these records into the tensor format required by TorchVision: RGB image, float32 boxes (N x 4), int64 labels, image ID, object area and iscrowd.')
add_table([['Split','Images','Number of objects'],['Train','150',str(sum(class_counts.values()))],['Validation','30','63'],['Test','30',str(metrics['ground_truth_instances'])]],widths=[169,169,170])
add_table([['Class','Training instances'],['Circle',str(class_counts['circle'])],['Square',str(class_counts['square'])],['Triangle',str(class_counts['triangle'])]],widths=[254,254])
fig('ground_truth_examples.png',440,maxheight=295)
caption('Figure 1. Generated training images with exact ground-truth bounding boxes and class labels.')
fig('class_distribution.png',360,maxheight=180)
caption('Figure 2. Training object-count distribution across the three detection classes.')
H('5. Model configuration and training')
para('The shared CNN backbone consists of four convolutional layers with output channels 24, 48, 64 and 64, and outputs a 64-channel feature map. Its output stride is 4. The RPN uses anchor sizes 16, 24, 32, 48 and 64 pixels with aspect ratios 0.75, 1.0 and 1.33. MultiScaleRoIAlign pools proposals into 4 x 4 feature grids. A 128-unit two-layer MLP feeds classification and bounding-box regression. The classifier predicts four labels including background. The network is trained from scratch (no pretrained weights).')
para('The training objective is the sum of the RPN objectness loss, RPN box regression loss, RoI classification loss and final box regression loss. We minimize it with Adam, learning rate 0.001 and gradient clipping at norm 5. Training uses horizontal flip augmentation (probability 0.5). Each epoch is evaluated on the validation split; the model with highest validation mAP@0.50 is retained.')
add_table([['Parameter','Setting'],['Random seed','42'],['Training images','150'],['Epochs','12'],['Batch size','4'],['Learning rate','0.001'],['Augmentation','Random horizontal flip on train split only'],['Compute','CPU, 4 threads in recorded run'],['Recorded training time',f"{summary['training_seconds']:.1f} seconds"]],widths=[210,298])
fig('training_curves.png',456,maxheight=220)
caption('Figure 3. Optimization loss and held-out validation mAP@0.50 across training epochs.')
H('6. Test metrics and evaluation protocol')
para('The model is selected using validation performance only; final metrics are measured on 30 previously unseen test images. A detection counts as a true positive if it has the correct class and its Intersection over Union (IoU) with a previously unmatched ground-truth box exceeds the relevant threshold. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. Average precision (AP) is the area under the interpolated precision-recall curve; mAP is averaged across the three classes. Precision, recall and F1 below use confidence >= 0.35; mAP uses ranked detections without imposing that display threshold.')
add_table([['Metric','Actual result'],['mAP@IoU 0.50',f"{metrics['mAP50']:.4f} ({metrics['mAP50']*100:.1f}%)"],
           ['mAP@IoU 0.75',f"{metrics['mAP75']:.4f} ({metrics['mAP75']*100:.1f}%)"],
           ['Precision / Recall / F1',f"{metrics['precision_at_threshold']:.4f} / {metrics['recall_at_threshold']:.4f} / {metrics['f1_at_threshold']:.4f}"],
           ['Mean IoU of matched detections',f"{metrics['mean_matched_iou']:.4f}"],
           ['True positives / false positives / false negatives',f"{metrics['true_positive']} / {metrics['false_positive']} / {metrics['false_negative']}"],
           ['Test images and annotated objects',f"{metrics['evaluated_images']} images / {metrics['ground_truth_instances']} objects"]],widths=[254,254])
add_table([['Class','AP50','AP75'],*[[k,f"{metrics['AP50_per_class'][k]:.4f}",f"{metrics['AP75_per_class'][k]:.4f}"] for k in ('circle','square','triangle')]],widths=[170,169,169])
fig('test_predictions.png',300,maxheight=445)
caption('Figure 4. Inference on independent test examples. Dotted outlines show ground truth; solid outlines show predictions with confidence scores.')
H('7. Interpretation and limitations')
para('The model correctly identified all 55 annotated objects in the test set at IoU 0.50 and 0.75, with no false positives at confidence 0.35. Its mean matched IoU of approximately 0.912 indicates tight predicted boxes. However, this near-perfect result is expected to be easier than a real visual detection benchmark: colors are strongly correlated with labels, objects are geometrically regular, image size is fixed, backgrounds are simple and there is little overlap or occlusion. This report does <b>not</b> claim that performance transfers to natural traffic signs, pedestrians or other photographs.')
para('A useful next experiment is to train the same pipeline on a separately annotated, real multi-scene dataset such as Penn-Fudan pedestrians or Pascal VOC, adding greater background diversity and evaluation at multiple IoU thresholds. Training from scratch on only 150 images is a controlled educational demonstration, not a production-ready detector.')
H('8. Reproduction and deliverables')
para('Inside the ZIP, run <b>pip install -r requirements.txt</b> once. Then run <b>python lab08.py all --seed 42 --n-train 150 --n-val 30 --n-test 30 --epochs 12</b>. This generates data, training plots, a trained checkpoint, test metrics and prediction images. For arbitrary image inference, run <b>python lab08.py predict --image PATH_TO_IMAGE</b>. The executable notebook <b>Lab08_Faster_RCNN.ipynb</b> mirrors all steps; saved model weights and already measured results are supplied to avoid requiring a rerun.')
para('<b>Delivered artifacts:</b> complete labeled train/val/test datasets (PNG and CSV), implementation script, notebook, model checkpoint, training log and metrics (JSON/CSV), prediction CSV, images with ground-truth and predicted boxes, and this report. The generated dataset is included, so no dataset download or API key is required.')
H('9. References')
story.append(P('1. Lab 08 Manual, Computer Vision AI-4002, sections 2–6 (provided assignment); task statement on page 17.','SmallReport'))
story.append(P('2. PyTorch / TorchVision, Object Detection Finetuning Tutorial. https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html','SmallReport'))
story.append(P('3. TorchVision documentation, Faster R-CNN and anchor generation. https://pytorch.org/vision/stable/models/faster_rcnn.html','SmallReport'))

pdf=SimpleDocTemplate(str(OUTPUT),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=54,bottomMargin=56,
                     title='Lab 08 - Faster R-CNN Object Detection',author='Computer Vision Lab Submission',subject='Fully reproducible Faster R-CNN experiment')
pdf.build(story,onFirstPage=page,onLaterPages=page)
print('Built',OUTPUT)
