import math

default_config = {'dataset_control_path': 'Data/control',
                  'dataset_study_path': 'Data/study',

                  'refresh_cache': False,  # Assign to True if looking to create a new cache
                  'dataset_limit': 2,  # Control # of samples in a newly created cache

                  'input_size': (496, 1024),  # Some samples are (496, 512)
                  'model_name': 'vgg19_nominal',
                  'epochs': 7,
                  'lr': 1e-4,
                  'batch_size': 30,

                  'device': 'cuda',
                  }
