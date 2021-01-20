import os
from datetime import datetime
from sklearn.ensemble import RandomTreesEmbedding
import h5py
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.models import Model


FILTER_SIZES = [
    [(9, 9), (5, 5), (3, 3)]
]

NO_FILTERS = [
    [32, 16, 8],
]


INPUT_DATASET = './dataset_191101_1224.hdf5'

# =====================================================================================================
# Load the data.
f = h5py.File(INPUT_DATASET, 'r')

depth_train = np.expand_dims(np.array(f['train/depth_inpainted']), -1)
point_train = np.expand_dims(np.array(f['train/grasp_points_img']), -1)
angle_train = np.array(f['train/angle_img'])
cos_train = np.expand_dims(np.cos(2*angle_train), -1)
sin_train = np.expand_dims(np.sin(2*angle_train), -1)
grasp_width_train = np.expand_dims(np.array(f['train/grasp_width']), -1)

depth_test = np.expand_dims(np.array(f['test/depth_inpainted']), -1)
point_test = np.expand_dims(np.array(f['test/grasp_points_img']), -1)
angle_test = np.array(f['test/angle_img'])
cos_test = np.expand_dims(np.cos(2*angle_test), -1)
sin_test = np.expand_dims(np.sin(2*angle_test), -1)
grasp_width_test = np.expand_dims(np.array(f['test/grasp_width']), -1)

# Ground truth bounding boxes.
gt_bbs = np.array(f['test/bounding_boxes'])

f.close()

# ====================================================================================================
# Set up the train and test data.

x_train = depth_train

grasp_width_train = np.clip(grasp_width_train, 0, 150)/150.0
y_train = [point_train, cos_train, sin_train, grasp_width_train, depth_train]

x_test = depth_test

grasp_width_test = np.clip(grasp_width_test, 0, 150)/150.0
y_test = [point_test, cos_test, sin_test, grasp_width_test, depth_test]

# ======================================================================================================================
# Eforest added by Remo
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
model = RandomTreesEmbedding(n_estimators=500, max_depth=None, n_jobs=-1)
print("Fitting forest")
#model.fit(x_train)
print("Encoding")
x_encoded_train = model.encode(x_train)
print("Decoding")
x_decoded_train = model.decode(x_encoded_train)

x_encoded_test = model.encode(x_test)
x_decoded_test = model.decode(x_encoded_test)

x_decoded_train = np.reshape(x_decoded_train,(len(x_train),300,300,1))
x_decoded_test = np.reshape(x_decoded_test,(len(x_test),300,300,1))


for filter_sizes in FILTER_SIZES:
    for no_filters in NO_FILTERS:
        dt = datetime.now().strftime('%y%m%d_%H%M')
        
        NETWORK_NAME = "ggcnn_%s_%s_%s__%s_%s_%s" % (filter_sizes[0][0], filter_sizes[1][0], filter_sizes[2][0],
                                                     no_filters[0], no_filters[1], no_filters[2])
        NETWORK_NOTES = """
            Input: Inpainted depth, subtracted mean, in meters, with random rotations and zoom. 
            Output: q, cos(2theta), sin(2theta), grasp_width in pixels/150.
            Dataset: %s
            Filter Sizes: %s
            No Filters: %s
        """ % (
            INPUT_DATASET,
            repr(filter_sizes),
            repr(no_filters)
        )
        OUTPUT_FOLDER = 'data/networks/%s__%s/' % (dt, NETWORK_NAME)
        
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        
        # Save the validation data so that it matches this network.
        np.save(os.path.join(OUTPUT_FOLDER, '_val_input'), x_test)
        
        # ====================================================================================================
        # Network
        
        input_layer = Input(shape=x_train.shape[1:])
        
        # ===================================================================================================
        # Output layers
        
        pos_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='pos_out')(input_layer)
        cos_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='cos_out')(input_layer)
        sin_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='sin_out')(input_layer)
        width_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='width_out')(input_layer)
        rec_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='rec_out')(input_layer)
        
        # ===================================================================================================
        # And go!
        
        # added by hamidreza
        ae = Model(input_layer, [pos_output, cos_output, sin_output, width_output, rec_output])
        
        ae.compile(optimizer='rmsprop', loss='mean_squared_error')
        
        ae.summary()
        
        with open(os.path.join(OUTPUT_FOLDER, '_description.txt'), 'w') as f:
            # Write description to file.
            f.write(NETWORK_NOTES)
            f.write('\n\n')
            ae.summary(print_fn=lambda q: f.write(q + '\n'))
        
        with open(os.path.join(OUTPUT_FOLDER, '_dataset.txt'), 'w') as f:
            # Write dataset name to file for future reference.
            f.write(INPUT_DATASET)
        
        tb_logdir = './data/tensorboard/%s_%s' % (dt, NETWORK_NAME)
        
        my_callbacks = [
            TensorBoard(log_dir=tb_logdir),
            ModelCheckpoint(os.path.join(OUTPUT_FOLDER, 'epoch_{epoch:02d}_model.hdf5'), period=1),
        ]
        
        ae.fit(x_decoded_train, y_train,
               batch_size=4,
               epochs=50,
               shuffle=True,
               callbacks=my_callbacks,
               validation_data=(x_decoded_test, y_test)
               )
