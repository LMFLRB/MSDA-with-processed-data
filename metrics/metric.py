
def get_correct(preds, labels):
    if len(preds.shape)>1 and preds.shape[1]>1:
        preds = preds.data.max(1)[1]
    return preds.eq(labels.data).cpu().sum()

def acc(preds, labels):
    return 100.*get_correct(preds, labels)/len(preds)