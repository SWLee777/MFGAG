from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import torch

class Recorder():
    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {v: k for k, v in category_dict.items()}

    for c in reverse_category_dict.values():
        res_by_category[c] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        if len(set(res['y_true'])) > 1:  # Ensure category has different labels
            metrics_by_category[c] = {
                'auc': roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        else:
            metrics_by_category[c] = {
                'auc': None  # Set to None if only one label in category
            }

    overall_metrics = {}
    if len(set(y_true)) > 1:  # Ensure overall has different labels
        overall_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    else:
        overall_metrics['auc'] = None

    y_pred = np.around(np.array(y_pred)).astype(int)
    overall_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    overall_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    overall_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    overall_metrics['acc'] = accuracy_score(y_true, y_pred)

    for c, res in res_by_category.items():
        y_pred_cat = np.around(np.array(res['y_pred'])).astype(int)
        metrics_by_category[c].update({
            'precision': precision_score(res['y_true'], y_pred_cat, average='macro', zero_division=0).round(4).tolist(),
            'recall': recall_score(res['y_true'], y_pred_cat, average='macro', zero_division=0).round(4).tolist(),
            'fscore': f1_score(res['y_true'], y_pred_cat, average='macro', zero_division=0).round(4).tolist(),
            'acc': accuracy_score(res['y_true'], y_pred_cat).round(4)
        })

    metrics_by_category.update(overall_metrics)
    return metrics_by_category

def domain_metrics(y_true, y_pred):
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics = {
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'fscore': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'acc': accuracy_score(y_true, y_pred)
    }
    return metrics

def data2gpu(batch, use_cuda):
    batch_data = {
        'content': batch['content_token_ids'].cuda(),
        'content_masks': batch['content_masks'].cuda(),
        'comments': batch['comments_token_ids'].cuda(),
        'comments_masks': batch['comments_masks'].cuda(),
        'content_emotion': batch['content_emotion'].cuda(),
        'comments_emotion': batch['comments_emotion'].cuda(),
        'emotion_gap': batch['emotion_gap'].cuda(),
        'style_feature': batch['style_feature'].cuda(),
        'label': batch['label'].cuda(),
        'category': batch['category'].cuda(),
        'edge_indices': batch['edge_indices'].cuda() if isinstance(batch['edge_indices'], torch.Tensor)
                       else batch['edge_indices'].cuda(),  # Modify edge_indices handling
        'ids': batch['ids'].cuda()  # Ensure ids are also transferred to GPU
    }
    return batch_data

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
