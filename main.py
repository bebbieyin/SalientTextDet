import time
import warnings
from fileinput import filename
import os

from tqdm import tqdm
import torch
from torchvision import transforms

from modeling.model import build_salientText_model
from modeling.optimizer import build_optimizer
from modeling.scheduler import build_scheduler
from modeling.data.dataloader import build_dataloader
from modeling.loss import bce_ohem, KL_loss
from modeling.metrics import precision, recall, iou_score, f1_score
from utils import *

warnings.filterwarnings("ignore")

def train_epoch(train_dataloader, model, optimizer, scheduler, device, thresh, epoch):

    train_loss = AverageMeter()
    results_f1 = AverageMeter()
    results_precision = AverageMeter()
    results_recall = AverageMeter()
    results_iou = AverageMeter()
    model.train()

    for idx, (_,images,saliency_gt,text_gt) in enumerate(tqdm(train_dataloader)):
        
        optimizer.zero_grad()
        
        images = images.to(device)
        saliency_gt = saliency_gt.to(device)
        text_gt = text_gt.to(device)
        gt_text = text_gt.round().long()

        pred_saliency, pred_text = model(images)

        saliency_loss = KL_loss(pred_saliency, saliency_gt)
        text_loss = bce_ohem(pred_text,text_gt)

        loss = saliency_loss+ text_loss 

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(),images.size(0))
        results_f1.update(f1_score(pred_text>thresh,gt_text>thresh).item())
        results_precision.update(precision(pred_text>thresh,gt_text>thresh).item())
        results_recall.update(recall(pred_text>thresh,gt_text>thresh).item())
        results_iou.update(iou_score(pred_text>thresh,gt_text>thresh).item())

    scheduler.step_update(epoch * len(train_data_loader) + idx)

    return model, optimizer, scheduler, train_loss, results_f1, results_precision, results_recall, results_iou


def test(valid_dataloader,model, thresh):

    test_loss = AverageMeter()
    results_f1 = AverageMeter()
    results_precision = AverageMeter()
    results_recall = AverageMeter()
    results_iou = AverageMeter()

    model.eval()
    for _, (_,images,saliency_gt,text_gt) in enumerate(tqdm(valid_dataloader)):

        with torch.no_grad():
            images = images.to(device)
            saliency_gt = saliency_gt.to(device)
            text_gt = text_gt.to(device)

            gt_text = text_gt.round().long()

            pred_saliency, pred_text = model(images)

            saliency_loss = KL_loss(pred_saliency, saliency_gt)
            text_loss = bce_ohem(pred_text,text_gt)
            loss = saliency_loss+text_loss

            test_loss.update(loss.item(),images.size(0))
            results_f1.update(f1_score(pred_text>thresh,gt_text>thresh).item())
            results_precision.update(precision(pred_text>thresh,gt_text>thresh).item())
            results_recall.update(recall(pred_text>thresh,gt_text>thresh).item())
            results_iou.update(iou_score(pred_text>thresh,gt_text>thresh).item())

    return test_loss, results_f1, results_precision, results_recall, results_iou
  
if __name__ == '__main__':


    config = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    preprocess =transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config['INPUT']['SIZE']),
    ])

    # load data
    train_data_loader, test_data_loader = build_dataloader(config=config, augmentation=preprocess)

    # load model
    model = build_salientText_model(backbone_cfg=config['MODEL']['BACKBONE_CFG'], input_size=config['INPUT']['SIZE'], device=device).to(device)

    optimizer = build_optimizer(config,model)
    scheduler = build_scheduler(config,optimizer,len(train_data_loader))

    # load checkpoint 
    start_epoch = 0
    if config['TRAIN']['PRETRAINED']:
        model, optimizer, scheduler, start_epoch = load_checkpoint(config['TRAIN']['PRETRAINED'], model, optimizer, scheduler)
        start_epoch +=1
    print("Starting from epoch %d"%start_epoch)

    # create logs 
    fieldnames = ['Epoch','Train_loss','Test_loss','Train_Precision', 'Test_Precision', 
                  'Train_Recall','Test_Recall', 'Train_F1', 'Test_F1','Train_IOU', 'Test_IOU']
   
    output_path = config['OUTPUT']['OUT_PATH']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path,config['OUTPUT']['LOGS_FILENAME'])
    create_logs(fieldnames=fieldnames, filepath=filename)

    highest_f1 = 0
    lowest_loss = 1e10
    best_epoch = 0

    start_time = time.time()

    # start training 
    for i in range(start_epoch, config['TRAIN']['EPOCHS']):
        torch.cuda.empty_cache()
        
        results = dict()

        print("Epoch %d/%d" %(i, config['TRAIN']['EPOCHS']))

        model, optimizer, scheduler, training_loss, train_f1, train_precision, train_recall, train_iou \
              = train_epoch(train_data_loader, model,optimizer,scheduler,device,config['OUTPUT']['PROB_THRESH'],i)

        test_loss, test_f1, test_precision, test_recall, test_iou = test(test_data_loader, model, config['OUTPUT']['PROB_THRESH'])

        results[fieldnames[0]] = i
        results[fieldnames[1]] = round(training_loss.avg,4)
        results[fieldnames[2]] = round(test_loss.avg,4)
        results[fieldnames[3]] = round(train_precision.avg,4)
        results[fieldnames[4]] = round(test_precision.avg,4)
        results[fieldnames[5]] = round(train_recall.avg,4)
        results[fieldnames[6]] = round(test_recall.avg,4)
        results[fieldnames[7]] = round(train_f1.avg,4)
        results[fieldnames[8]] = round(test_f1.avg,4)
        results[fieldnames[9]] = round(train_iou.avg,4)
        results[fieldnames[10]] = round(test_iou.avg,4)

        print(results)

        if test_f1.avg > highest_f1:
            highest_f1 = round(test_f1.avg,4)
            save_checkpoint(config,i,model,optimizer,scheduler)
            best_epoch = i
        
        if test_loss.avg < lowest_loss:
            lowest_loss = round(test_loss.avg,4)
            save_checkpoint(config,i,model,optimizer,scheduler)

        print("Best Epoch: %d"%best_epoch)
        print("Best F1: %f"%highest_f1)
        print("Best Loss: %f"%lowest_loss)
        print("\n")


        save_results(csv_path = filename,results = results, fieldnames=fieldnames)
    print("Training time: %.2f mins" % (round((time.time() - start_time) / 60, 2)))


