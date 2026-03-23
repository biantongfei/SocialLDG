import os


def get_first_id(feature_json):
    for frame in feature_json['frames']:
        return frame['frame_id']
    return -1


def get_autoencoder_pose_tra_test_files(data_path):
    tra_files = []
    val_files = []
    test_files = []

    for dataset_name in ['JPL_Social', 'HARPER']:
        jpl_data_path = data_path + dataset_name
        tra_files += [jpl_data_path + '/train/' + i for i in os.listdir(jpl_data_path + '/train/') if
                      i.endswith('.json')]
        val_files += [jpl_data_path + '/validation/' + i for i in os.listdir(jpl_data_path + '/validation/') if
                      i.endswith('.json')]
        test_files += [jpl_data_path + '/test/' + i for i in os.listdir(jpl_data_path + '/test/') if
                       i.endswith('.json')]

    return tra_files, val_files, test_files
