# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
#%matplotlib inline

import sys
project_root = '/home/chigo/working/research/faster-rcnn/py-faster-rcnn-master/'
sys.path.insert(0, project_root + 'caffe-fast-rcnn/python')    #caffe python
import caffe, cv2, os
import argparse

keyfile = '/home/chigo/working/research/faster-rcnn/py-faster-rcnn-master/python-visualization/keyfile/'
model_def = keyfile + 'imagenet/deploy_imagenet.prototxt'
model_weights = keyfile + 'imagenet/bvlc_reference_caffenet.caffemodel'
model_means = keyfile + 'imagenet/ilsvrc_2012_mean.npy'
labels_file = keyfile + 'imagenet/ilsvrc_2012_labels.txt'

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

def vis_square(data, svFileName):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.clf()
    plt.imshow(data)
    plt.axis('off')
    plt.savefig(svFileName)

def demo_oneimage_mutilabel(net, im_file, output_path, im_name):
    """Detect object classes in an image using pre-computed object proposals."""

    image = caffe.io.load_image(im_file)
    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image)

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()
    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
    print 'predicted class is:', output_prob.argmax()
    
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    print 'output label:', labels[output_prob.argmax()]

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
    print 'probabilities and labels:',zip(output_prob[top_inds], labels[top_inds])

    # the parameters are a list of [weights, biases]
    filters = net.params['conv1'][0].data
    output_file = os.path.join( output_path, 'conv1_kernal.png' )  
    vis_square(filters.transpose(0, 2, 3, 1), output_file)

    feat = net.blobs['conv1'].data[0]
    output_file = os.path.join( output_path, 'conv1.png' )
    vis_square(feat, output_file)

    feat = net.blobs['pool1'].data[0]
    output_file = os.path.join( output_path, 'pool1.png' )
    vis_square(feat, output_file)

    feat = net.blobs['norm1'].data[0]
    output_file = os.path.join( output_path, 'norm1.png' )
    vis_square(feat, output_file)

    feat = net.blobs['conv2'].data[0]
    output_file = os.path.join( output_path, 'conv2.png' )
    vis_square(feat, output_file)

    feat = net.blobs['pool2'].data[0]
    output_file = os.path.join( output_path, 'pool2.png' )
    vis_square(feat, output_file)

    feat = net.blobs['norm2'].data[0]
    output_file = os.path.join( output_path, 'norm2.png' )
    vis_square(feat, output_file)

    feat = net.blobs['conv3'].data[0]
    output_file = os.path.join( output_path, 'conv3.png' )
    vis_square(feat, output_file)

    feat = net.blobs['conv4'].data[0]
    output_file = os.path.join( output_path, 'conv4.png' )
    vis_square(feat, output_file)

    feat = net.blobs['conv5'].data[0]
    output_file = os.path.join( output_path, 'conv5.png' )
    vis_square(feat, output_file)

    feat = net.blobs['pool5'].data[0]
    output_file = os.path.join( output_path, 'pool5.png' )
    vis_square(feat, output_file)

    featlist = net.blobs['fc6'].data[0]
    plt.clf()
    plt.subplot(7, 1, 1)
    plt.plot(featlist.flat)
    plt.subplot(7, 1, 2)
    _ = plt.hist(featlist.flat[featlist.flat > 0], bins=100)

    featlist = net.blobs['fc7'].data[0]
    plt.subplot(7, 1, 3)
    plt.plot(featlist.flat)
    plt.subplot(7, 1, 4)
    _ = plt.hist(featlist.flat[featlist.flat > 0], bins=100)

    featlist = net.blobs['fc8'].data[0]
    plt.subplot(7, 1, 5)
    plt.plot(featlist.flat)
    plt.subplot(7, 1, 6)
    _ = plt.hist(featlist.flat[featlist.flat > 0], bins=100)

    featlist = net.blobs['prob'].data[0]
    plt.subplot(7, 1, 7)
    plt.plot(featlist.flat)
    output_file = os.path.join( output_path, 'fc6_fc7_fc8_prob.png' )     
    plt.savefig(output_file)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--input', dest='input_file', default='/home/chigo/working/research/faster-rcnn/py-faster-rcnn-master/python-visualization/img/imglist.txt')
    parser.add_argument('--output', dest='output_path', default='/home/chigo/working/research/faster-rcnn/py-faster-rcnn-master/python-visualization/sv_imagenet/')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    output_dir = args.output_path[:args.output_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = open(args.input_file, "r")
    alllines = file.readlines();
    file.close();

    if os.path.isfile(model_weights):
        print 'CaffeNet found.'

    caffe.set_mode_gpu()
    caffe.set_device(0)
    #caffe.set_mode_cpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(model_weights)

    mu = np.load(model_means)
    mu = mu.mean(1).mean(1)
    print 'mean-subtracted values:', zip('BGR', mu)

    # load ImageNet labels
    if not os.path.exists(labels_file):
        print 'err on:', labels_file

    # for each layer, show the output shape
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)

    for layer_name, param in net.params.iteritems():
        print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    net.blobs['data'].reshape(1, 3, 227, 227)
    nCount = 0;
    for line in alllines:

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'line:{}'.format(line)
        im_name = line[(line.rfind('/')+1):];
        im_iid = im_name[:im_name.rfind('.')];
        print 'im_iid:{}'.format(im_iid)
        im_name = '{:s}.jpg'.format(im_iid)
        input_path = line[:line.rfind('/')];
        im_file = '{:s}/{:s}'.format(input_path,im_name)
        print 'input_file:{}'.format(im_file)

        output_path = os.path.join( output_dir, im_iid )    
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        nCount = nCount+1;

        #add by chigo                    
        demo_oneimage_mutilabel(net, im_file, output_path, im_name)

    print 'All load img:{}!!'.format(nCount)

















