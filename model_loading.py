# Save Model, Load Model, Examine model properties
import pickle
import os
from datetime import datetime, date
import numpy as np
import tensorflow as tf, tf_keras
from tf_keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model


def load_movinet_backbone(model_id = 'a0', resolution = 172):

    use_positional_encoding = model_id in {'a3', 'a4', 'a5'}
    backbone = movinet.Movinet(
        model_id=model_id,
        causal=True,
        conv_type='2plus1d',
        se_type='2plus3d',
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        use_positional_encoding=use_positional_encoding,
        use_external_states=False,
    )

    return backbone

def detect_hardware_distribution():
    try:
        tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    except ValueError:
        tpu_resolver = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy
    if tpu_resolver:
        tf.config.experimental_connect_to_cluster(tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
        distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
        print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
    elif len(gpus) > 1:
        distribution_strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        distribution_strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on single GPU ', gpus[0].name)
    else:
        distribution_strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on CPU')

    print("Number of accelerators: ", distribution_strategy.num_replicas_in_sync)

    return tpu_resolver, distribution_strategy

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model

def load_seg_model_weights(model_specs, loss_obj = None, optimizer = None):

    backbone = load_movinet_backbone(model_id= model_specs['model_id'], resolution = model_specs['resolution'])
    _, distribution_strategy = detect_hardware_distribution()
    # Construct loss, optimizer and compile the model
    with distribution_strategy.scope():
        model = build_classifier(model_specs['batch_size'], model_specs['num_frames'], model_specs['resolution'], backbone, model_specs['num_classes'])
        if loss_obj is None:
            if ((model_specs['num_classes'] <= 2) & (model_specs['loss_name'] is None)) or (model_specs['loss_name'] == 'BinaryCrossentropy'):
                loss_obj =  BinaryCrossentropy(from_logits=True)
                loss_name = 'BinaryCrossentropy'
            elif ((model_specs['num_classes'] > 2) & (model_specs['loss_name'] is None)) or (model_specs['loss_name'] == 'SparseCategoricalCrossentropy'):
                loss_obj = SparseCategoricalCrossentropy(from_logits = True)
                loss_name = 'SparseCategoricalCrossentropy'

        if optimizer is None:
            optimizer = tf_keras.optimizers.legacy.Adam(learning_rate = model_specs['learning_rate'])

        model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

        if model_specs['checkpoint_path'] is None:
            model.load_weights(tf.train.latest_checkpoint(model_specs['checkpoint_dir']))
        else:
            model.load_weights(model_specs['checkpoint_path'])

        return model

def load_seg_model_data(model_path):

    # Changed it to not actually

    files = [x for x in os.listdir(model_path) if '_md_' in x]
    if len(files) > 1:
        target_file = np.argmax([datetime.strptime(x.split('_')[1], "%m%d%y").date() for x in files])
        model_fn = os.path.join(model_path, files[target_file])
    elif len(files) == 1:
        model_fn = os.path.join(model_path, files[0])
    else:
        print('No model file in folder')
        return None, None, None

    with open(model_fn, 'rb') as file:
        model_data = pickle.load(file)

    return model_data['model_specs'], model_data['data_specs']

def create_seg_model(model_id, resolution, batch_size, num_frames, num_classes, loss_name, optimizer = None, loss_obj = None, learning_rate = 0.001):
    backbone = load_movinet_backbone(model_id= model_id, resolution = resolution)
    _, distribution_strategy = detect_hardware_distribution()
    # Construct loss, optimizer and compile the model
    with distribution_strategy.scope():
        model = build_classifier(batch_size, num_frames, resolution, backbone, num_classes)
        if loss_obj is None:
            if ((num_classes == 1) & (loss_name is None)) or (loss_name == 'BinaryCrossentropy'):
                loss_obj =  BinaryCrossentropy(from_logits=True)
                loss_name = 'BinaryCrossentropy'
            elif ((num_classes > 1) & (loss_name is None)) or (loss_name == 'SparseCategoricalCrossentropy'):
                loss_obj = SparseCategoricalCrossentropy(from_logits = True)
                loss_name = 'SparseCategoricalCrossentropy'

        if optimizer is None:
            optimizer = tf_keras.optimizers.legacy.Adam(learning_rate = learning_rate)

        model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

    model_specs = { 'checkpoint_dir': None, 'checkpoint_path': None,  'model_id':model_id, 'resolution': resolution, 'batch_size': batch_size, 
                   'num_frames': num_frames, 'num_classes': num_classes, 'loss_name': loss_name, 'learning_rate': learning_rate}
    
    return model, model_specs

def save_segmentation_model(save_path, model_specs, data_specs, accuracy, loss, test_train_split):

    model_data = {'model_specs': model_specs, 'data_specs': data_specs}
    training_data = {'accuracy': accuracy, 'loss': loss}# , 'test_train_split': test_train_split}

    today = date.today().strftime('%m%d%y')
    model_name = os.path.basename(model_specs['checkpoint_dir']).split('_checkpoints')[0]

    model_fn = os.path.join(save_path, model_name + "_md_" + today + ".pkl")
    training_fn = os.path.join(save_path, model_name + "_tr_" + today + ".pkl")
    with open(model_fn, 'wb') as file:
        pickle.dump(model_data, file)

    with open(training_fn, 'wb') as file:
        pickle.dump(training_data, file)
    
    tr_fn = os.path.join(save_path, model_name+ '_training_sets', model_name + "_" + today + ".pkl")

    with open(tr_fn, 'wb') as file:
        pickle.dump(test_train_split, file)

def view_segmentation_model(model_path):

    files = [x for x in os.listdir(model_path) if '_tr_' in x]
    if len(files) > 1:
        target_file = np.argmax([datetime.strptime(x.split('_')[1], "%m%d%y").date() for x in files])
        train_fn = os.path.join(model_path, files[target_file])
    elif len(files) == 1:
        train_fn = os.path.join(model_path, files[0])
    else:
        print('No training file in folder')
        return None, None, None

    with open(train_fn, 'rb') as file:
        train_data = pickle.load(file)

    files = [x for x in os.listdir(model_path) if '_md_' in x]
    if len(files) > 1:
        target_file = np.argmax([datetime.strptime(x.split('_')[1], "%m%d%y").date() for x in files])
        model_fn = os.path.join(model_path, files[target_file])
    elif len(files) == 1:
        model_fn = os.path.join(model_path, files[0])
    else:
        print('No model file in folder')
        return None, None, None

    with open(model_fn, 'rb') as file:
        model_data = pickle.load(file)

    return train_data, model_data['model_specs'], model_data['data_specs']



# necessary components of each dict
# model_specs = [checkpoint_dir, checkpoint_path, model_id, resolution, batch_size, num_frames, num_classes, loss_name, learning_rate, **class_labels]
# data_specs = [num_frames, frame_step, buffer, output_size]
# crop_specs = [dp, minDist, animals]
# training_data = [accuracy: dict of val, train, test accs, loss: dict of val, train, test vals, test_train_split: table used to generate training set]