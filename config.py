import math

default_config = {'dataset_control_path': 'Data/control',
                  'dataset_study_path': 'Data/study',

                  'refresh_cache': True,  # Assign to True if looking to create a new cache
                  'control_limit': math.inf, 'study_limit': math.inf,  # Control # of samples in a newly created cache

                  'input_size': (496, 1024),  # Some samples are (496, 512)
                  'model_name': 'vgg19_nominal',
                  'epochs': 7,
                  'lr': 1e-4,
                  'batch_size': 50,

                  'device': 'cuda',
                  }
