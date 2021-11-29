import math

# Default configuration to use
from util import dot_dict

default_config = dot_dict({'dataset_control_path': 'Data/control',
                  'dataset_study_path': 'Data/study',

                  'refresh_cache': True,  # Assign to True if looking to create a new cache

                  'input_size': (496, 1024),  # Some samples are (496, 512)
                  'model_name': 'vgg19_nominal',  # 'resnet18_nominal'
                  'epochs': 7,
                  'lr': 1e-4,
                  'batch_size': 50,

                  'device': 'cuda',
                  })
