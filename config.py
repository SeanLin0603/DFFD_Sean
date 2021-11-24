
class Config:
        # dataRoot = 'D:\\Dataset\\DFFD\\data\\'
        # dataRoot = 'C:\\Users\\SeanVIP\\Documents\\FaceForensic\\align\\'
        dataRoot = '/home/sean/faceforensic/'

        saveDir = './models/'
        backbone = 'xcp'
        mapType = 'tmp'
        batch_size = 3
        maxEpochs = 9
        stepsPerEpoch = 10
        learningRate = 0.0001
        weightDecay = 0.01

        modelConfig = {
        'xcp': {
                'img_size': (299, 299),
                'map_size': (19, 19),
                'norms': [[0.5] * 3, [0.5] * 3]
                },
        'vgg': {
                'img_size': (299, 299),
                'map_size': (19, 19),
                'norms': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
                }
        }

