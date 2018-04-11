import coremltools as cmt
import sys, os.path as op

caffe_w = sys.argv[1]
name = op.splitext(caffe_w)[0]
caffe_p = name + '.prototxt'
caffe_l = name + '.labels'
print(caffe_w,caffe_p,caffe_l);
#mod = cmt.converters.caffe.convert((caffe_w,caffe_p),blue_bias=-104,green_bias=-117,red_bias=-123, image_input_names='data')
mod = cmt.converters.caffe.convert((caffe_w,caffe_p),model_precision='float16',blue_bias=-104,green_bias=-117,red_bias=-123, image_input_names='data')
mod.save(name+'.mlmodel')

