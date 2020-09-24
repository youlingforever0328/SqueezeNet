def fire_module(self, inputs, squeeze_depth, expand_depth, scope):
        with fluid.scope_guard(scope):
            squeeze =fluid.layers.conv2d(inputs, squeeze_depth, filter_size=1,
                                       stride=1, padding="VALID",
                                       act='relu', name="squeeze")
            #print('squeeze shape:',squeeze.shape)
            # squeeze
            expand_1x1 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=1,
                                          stride=1, padding="VALID",
                                          act='relu', name="expand_1x1")
            #print('expand_1x1 shape:',expand_1x1.shape)

            expand_3x3 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=3,
                                          stride=1, padding=1,
                                          act='relu', name="expand_3x3")
            #print('expand_3x3 shape:',expand_3x3.shape)
            return fluid.layers.concat([expand_1x1, expand_3x3], axis=1)

class SqueezeNet(object):
    def __init__(self, inputs, num_classes=1000):
        # conv_1
        net = fluid.layers.conv2d(inputs, 96, filter_size=7, stride=2,
                                 padding=2, act="relu",
                                 name="conv_1")
        
        # maxpool_1
        net = fluid.layers.pool2d(net, 3, 'max',2,name="maxpool_1")
        
        # fire2
        net = self.fire_module(net, 16, 64, "fire2")
        
        # fire3
        net = self.fire_module(net, 16, 64, "fire3")
        
        # fire4
        net = self.fire_module(net, 32, 128, "fire4")
        
        # maxpool_4
        net = fluid.layers.pool2d(net, pool_size=3, pool_type='max',pool_stride=2,name="maxpool_4")
        
        # fire5
        net = self.fire_module(net, 32, 128, "fire5")
        
        # fire6
        net = self.fire_module(net, 48, 192, "fire6")
        
        # fire7
        net = self.fire_module(net, 48, 192, "fire7")
        
        # fire8
        net = self.fire_module(net, 64, 256, "fire8")
        
        # maxpool_8
        net = fluid.layers.pool2d(net, pool_size=3, pool_type='max',pool_stride=2,name="maxpool_8")
        
        # fire9
        net = self.fire_module(net, 64, 256, "fire9")
        
        # dropout
        net = fluid.layers.dropout(net, 0.5)
        
        # conv_10
        net = fluid.layers.conv2d(net, num_classes, filter_size=1, stride=1,
                               padding="VALID", act=None,
                               name="conv_10")
        net = fluid.layers.batch_norm(net,act='relu')
        # avgpool_10
        net = fluid.layers.pool2d(net, pool_size=13, pool_type='avg',pool_stride=1,name="avgpool_10")
        
        # squeeze the axis
        net = fluid.layers.squeeze(net, axes=[2, 3])
        

        self.logits = net
        self.prediction = fluid.layers.softmax(net)

    def fire_module(self, inputs, squeeze_depth, expand_depth, scope):
        with fluid.scope_guard(scope):
            squeeze =fluid.layers.conv2d(inputs, squeeze_depth, filter_size=1,
                                       stride=1, padding="VALID",
                                       act='relu', name="squeeze")
            #print('squeeze shape:',squeeze.shape)
            # squeeze
            expand_1x1 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=1,
                                          stride=1, padding="VALID",
                                          act='relu', name="expand_1x1")
            #print('expand_1x1 shape:',expand_1x1.shape)

            expand_3x3 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=3,
                                          stride=1, padding=1,
                                          act='relu', name="expand_3x3")
            #print('expand_3x3 shape:',expand_3x3.shape)
            return fluid.layers.concat([expand_1x1, expand_3x3], axis=1)