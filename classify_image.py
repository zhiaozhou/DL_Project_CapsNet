import tensorflow as tf
import numpy as np
import glob
from PIL import Image
import cv2

def run_inference_on_image(filename):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(glob.glob('checkpoint_CapsNet/*.meta')[0])
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_CapsNet/'))

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name('pow_13:0')
    x = graph.get_tensor_by_name('Placeholder:0')
    on_train = graph.get_tensor_by_name('Placeholder_2:0')
    
    image=cv2.imread(filename)
    res=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(filename, res)
    pixels = np.asarray(Image.open(filename))
    if pixels.shape[-1] >= 3:
        feed_dict = {x:pixels[:,:,:3].reshape(-1,28,28,3)/255.,on_train:False}
    else:
        pixels = np.array([pixels[:,:,0]]*3)
        feed_dict = feed_dict = {x:pixels[:,:,:].reshape(-1,28,28,3)/255.,on_train:False}
        
    pred = sess.run(y_pred,feed_dict)
    top_k = list(np.argsort(pred, axis=-1, kind='quicksort', order=None)[::-1][:5])
    top_names = list(sorted(pred,reverse=True)[:5])
    predictions = dict(zip(top_k,top_names))
    
    return predictions,top_k,top_names