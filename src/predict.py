import os, time, argparse
import cv2
import numpy as np
import collections

from torch.multiprocessing import Pool
from configs.CC import Config
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from utils.nms_wrapper import nms
from utils.timer import Timer
from utils.core import *
from utils.pycocotools.coco import COCO

from firebase_admin import storage, db

args = {
    'config': 'configs/m2det512_vgg.py',
    'trained_model': 'weights/m2det512_vgg.pth',
    'threshold': 0.3
}

def _to_color(index, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - index / base2
    r = 2 - (index % base2) / base
    g = 2 - (index % base2) % base
    return b * 127, r * 127, g * 127

def draw_detection(im, bboxes, scores, cls_inds, fps, colors, labels, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])

        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                        (box[0], box[1]), (box[2], box[3]),
                        colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)
    return imgcv

def download_image(gcp, user_id, timestamp):
    storage_client = gcp['client']
    bucket = storage_client.bucket(os.getenv('FIREBASE_BUCKET'))
    bucket_file = 'images/' + user_id + '/' + timestamp + '.jpeg'
    blob = bucket.blob(bucket_file)
    if not os.path.exists('/src/download/'):
        os.makedirs('/src/download/')
    if not os.path.exists('/src/download/' + user_id):
        os.makedirs('/src/download/' + user_id)
    local_file = '/src/download/' + user_id + '/' + timestamp + '.jpeg'
    blob.download_to_filename(local_file)

def main(gcp, user_id, timestamp):
    global cfg
    cfg = Config.fromfile(args['config'])
    anchor_config = anchors(cfg)
    # print_info('The Anchor info: \n{}'.format(anchor_config))
    priorbox = PriorBox(anchor_config)
    net = build_net('test',
                    size = cfg.model.input_size,
                    config = cfg.model.m2det_config)
    init_net(net, cfg, args['trained_model'])
    # print_info('===> Finished constructing and loading model',['yellow','bold'])
    net.eval()
    with torch.no_grad():
        priors = priorbox.forward()
        if cfg.test_cfg.cuda:
            net = net.cuda()
            priors = priors.cuda()
            cudnn.benchmark = True
        else:
            net = net.cpu()
    net = net.cpu()
    _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
    detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)

    base = int(np.ceil(pow(cfg.model.m2det_config.num_classes, 1. / 3)))
    colors = [_to_color(x, base) for x in range(cfg.model.m2det_config.num_classes)]
    cats = [_.strip().split(',')[-1] for _ in open('data/coco_labels.txt','r').readlines()]
    labels = tuple(['__background__'] + cats)

    download_image(gcp, user_id, timestamp)
    im_path = '/src/download/' + user_id

    im_fnames = sorted((fname for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] == '.jpeg'))
    im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
    im_iter = iter(im_fnames)
    while True:
        try:
            fname = next(im_iter)
        except StopIteration:
            break
        if 'm2det' in fname: continue # ignore the detected images
        image = cv2.imread(fname, cv2.IMREAD_COLOR)

        loop_start = time.time()
        w,h = image.shape[1],image.shape[0]
        img = _preprocess(image).unsqueeze(0)
        if cfg.test_cfg.cuda:
            img = img.cuda()
        scale = torch.Tensor([w,h,w,h])
        out = net(img)
        boxes, scores = detector.forward(out, priors)
        boxes = (boxes[0]*scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        for j in range(1, cfg.model.m2det_config.num_classes):
            inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            soft_nms = cfg.test_cfg.soft_nms
            keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist()+[j] for _ in c_dets])

        loop_time = time.time() - loop_start
        allboxes = np.array(allboxes)
        boxes = allboxes[:,:4]
        scores = allboxes[:,4]
        cls_inds = allboxes[:,5]
        # print('\n'.join(['pos:{}, ids:{}, score:{:.3f}'.format('(%.1f,%.1f,%.1f,%.1f)' % (o[0],o[1],o[2],o[3]) \
        #         ,labels[int(oo)],ooo) for o,oo,ooo in zip(boxes,cls_inds,scores)]))
        result, tmp_boxes, tmp_cls_ind, tmp_scores = [], [], [], []
        for box, cls_ind, score in zip(boxes,cls_inds,scores):
            if (score > args['threshold']):
                result.append(labels[int(cls_ind)])
                tmp_boxes.append(box)
                tmp_cls_ind.append(cls_ind)
                tmp_scores.append(score)
        boxes, cls_inds, scores = tmp_boxes, tmp_cls_ind, tmp_scores
        result = collections.Counter(result)
        
        # firebaseに検出結果を表示
        results_ref = db.reference('/results/' + user_id)
        results_ref.child(timestamp).update(dict(result))

        fps = -1
        im2show = draw_detection(image, boxes, scores, cls_inds, fps, colors, labels)

        if im2show.shape[0] > 1100:
            im2show = cv2.resize(im2show,
                                (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))

        cv2.imwrite('{}_m2det.jpeg'.format(fname.split('.')[0]), im2show)
        
        # gcsにアップロード
        storage_file_path = 'result/images/' + user_id + '/' + timestamp +'.jpeg'
        blob_gcs = gcp['bucket'].blob(storage_file_path)
        result_image_file = '{}_m2det.jpeg'.format(fname.split('.')[0])
        blob_gcs.upload_from_filename(result_image_file)

        if os.path.exists('/src/download/' + user_id):
            shutil.rmtree('/src/download/' + user_id)
            os.mkdir('/src/download/' + user_id)

if __name__ == "__main__":
    main()