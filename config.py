
class Config:
    saveDir = './models/'
    backbone = 'xcp'
    mapType = 'tmp'
    batch_size = 10
    maxEpochs = 1
    stepsPerEpoch = 630
    learningRate = 0.0001
    weightDecay = 0.001

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