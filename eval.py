import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.arrayprint import printoptions
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import auc

EPOCH = '0'
RESDIR = './models/xcp_tmp/results/' + EPOCH + '/'
# RESDIR = './models/results/' + EPOCH + '/'
RESFILENAMES = glob.glob(RESDIR + '*.mat')
MASK_THRESHOLD = 0.5

fileNums = len(RESFILENAMES)

print('{0} result files'.format(fileNums))

def compute_result_file(rfn):
  rf = loadmat(rfn)
  res = {}
  for r in ['lab', 'msk', 'score', 'pred', 'mask']:
    res[r] = rf[r].squeeze()
  return res

# Compile the results into a single variable for processing

rf = compute_result_file(RESFILENAMES[0])
TOTAL_RESULTS = {'lab': rf['lab'], 'msk': rf['msk'], 'score': rf['score'], 'pred': rf['pred'], 'mask': rf['mask']}

for i in range(fileNums):
  print("\rLoading...#{}/{}".format(i, fileNums), end=' ')
  rfn = RESFILENAMES[i] 
  rf = compute_result_file(rfn)

  rf['lab'] = rf['lab'].reshape((rf['lab'].size,))
  # print("[Info] rf['lab']:{}".format(rf['lab'].shape))
  # print("[Info] TOTAL_RESULTS['lab']:{}".format(TOTAL_RESULTS['lab'].shape))
  TOTAL_RESULTS['lab'] = np.concatenate((TOTAL_RESULTS['lab'], rf['lab']), axis=0)
  
  rf['msk'] = rf['msk'].reshape((int(rf['msk'].size/361), 19, 19))
  # print("[Info] rf['msk']:{}".format(rf['msk'].shape))
  # print("[Info] TOTAL_RESULTS['msk']:{}".format(TOTAL_RESULTS['msk'].shape))
  TOTAL_RESULTS['msk'] = np.concatenate((TOTAL_RESULTS['msk'], rf['msk']), axis=0)

  rf['score'] = rf['score'].reshape((int(rf['score'].size/2), 2))
  # print("[Info] rf['score']:{}".format(rf['score'].shape))
  # print("[Info] TOTAL_RESULTS['score']:{}".format(TOTAL_RESULTS['score'].shape))
  TOTAL_RESULTS['score'] = np.concatenate((TOTAL_RESULTS['score'], rf['score']), axis=0)
  
  rf['pred'] = rf['pred'].reshape((rf['pred'].size,))
  # print("[Info] rf['pred']:{}".format(rf['pred'].shape))
  # print("[Info] TOTAL_RESULTS['pred']:{}".format(TOTAL_RESULTS['pred'].shape))
  TOTAL_RESULTS['pred'] = np.concatenate((TOTAL_RESULTS['pred'], rf['pred']), axis=0)

  rf['mask'] = rf['mask'].reshape((int(rf['mask'].size/361), 19, 19))
  # print("[Info] rf['mask']:{}".format(rf['mask'].shape))
  # print("[Info] TOTAL_RESULTS['mask']:{}".format(TOTAL_RESULTS['mask'].shape))
  TOTAL_RESULTS['mask'] = np.concatenate((TOTAL_RESULTS['mask'], rf['mask']), axis=0)

print()
print('Found {0} total images with scores.'.format(TOTAL_RESULTS['lab'].shape[0]))
print('  {0} results are real images'.format((TOTAL_RESULTS['lab'] == 0).sum()))
print('  {0} results are fake images'.format((TOTAL_RESULTS['lab'] == 1).sum()))
#for r in TOTAL_RESULTS:
#  print('{0} has shape {1}'.format(r, TOTAL_RESULTS[r].shape))

# Compute the performance numbers
PRED_ACC = (TOTAL_RESULTS['lab'] == TOTAL_RESULTS['pred']).astype(np.float32).mean()
MASK_ACC = ((TOTAL_RESULTS['mask'] >= MASK_THRESHOLD) == (TOTAL_RESULTS['msk'] >= MASK_THRESHOLD)).astype(np.float32).mean()

FPR, TPR, THRESH = metrics.roc_curve(TOTAL_RESULTS['lab'], TOTAL_RESULTS['score'][:,1], drop_intermediate=False)
AUC = auc(FPR, TPR)
FNR = 1 - TPR
EER = FNR[np.argmin(np.absolute(FNR - FPR))]
TPR_AT_FPR_NOT_0 = TPR[FPR != 0].min()
TPR_AT_FPR_THRESHOLDS = {}
for t in range(-1, -7, -1):
  thresh = 10**t
  TPR_AT_FPR_THRESHOLDS[thresh] = TPR[FPR <= thresh].max()

# Print out the performance numbers
print('Prediction Accuracy: {0:.4f}'.format(PRED_ACC))
print('Mask Accuracy: {0:.4f}'.format(MASK_ACC))
print('AUC: {0:.4f}'.format(AUC))
print('EER: {0:.4f}'.format(EER))
print('Minimum TPR at FPR != 0: {0:.4f}'.format(TPR_AT_FPR_NOT_0))

print('TPR at FPR Thresholds:')
for t in TPR_AT_FPR_THRESHOLDS:
  print('  {0:.10f} TPR at {1:.10f} FPR'.format(TPR_AT_FPR_THRESHOLDS[t], t))

fig = plt.figure()
plt.plot(FPR, TPR)
plt.xlabel('FPR (%)')
plt.ylabel('TPR (%)')
plt.xscale('log')
plt.xlim([10e-8,1])
plt.ylim([0, 1])
plt.grid()
plt.show()
