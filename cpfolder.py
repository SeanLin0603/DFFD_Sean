import os, shutil
from pathlib import Path

srcDir = '/home/sean/Documents/faceforensic/frame/val/original'
maskDir = '/home/sean/Documents/faceforensic/frame/val/mask'
dstDir = '/home/sean/faceforensic/eval/real'


srcFolders = os.listdir(srcDir)
srcFolderNum = len(srcFolders)
print("[Info] srcFolderNum: {}".format(srcFolderNum))

maskFolders = os.listdir(maskDir)
maskFolderNum = len(maskFolders)
print("[Info] maskFolders: {}".format(maskFolderNum))
# print(maskFolders)

count = 0
for folder in maskFolders:
    cpFolder = os.path.join(srcDir, folder)
    dstFolder = os.path.join(dstDir, folder)
    # print(cpFolder)

    if not os.path.isdir(dstFolder):
        os.makedirs(dstFolder, mode=755)

    allFiles = os.listdir(cpFolder)
    # print(allFiles)

    for file in allFiles:
        cpFile = os.path.join(cpFolder, file)
        dstFile = os.path.join(dstFolder, file)
        print(cpFile)

        shutil.copy(cpFile, dstFile)
        print(dstFile)



