import os

from cpr import parse_yaml_config, CorrelationPatternRecognition, update_config

if __name__ == '__main__':
    # Read configuration file.
    CONFIG_PATH = os.path.join('data', 'config.yml')
    config = parse_yaml_config(CONFIG_PATH)
    basic_config, default_config = config['basic'], config['default']

    # Create main CPR object.
    cpr_object = CorrelationPatternRecognition(basic_config)

    # Create some different configuration using this template. You can use any keyword argument.
    # If you pass input_slm and/or filter_slm, you can use any of presented keys (other values will be set to default).
    config = update_config(
        basic_config,
        default_config,
        input_slm_type='amplitude',
        filter_slm_type='amplitude',
        filter_type='all',
        metrics=['peak', 'pce', 'cnn'],
        save_data=True,
        input_slm={
            'phase_depth': 2.0,
            'noise_type': 'amplitude-phase',
            'noise_level': 0.1,
            'phase_type': 'linear'
        },
        filter_slm={
            'sample_level': 8,
            'phase_depth': 2.0,
            'noise_type': None,
            'noise_level': 0.0,
            'phase_type': 'random'
        },
        use_same_images_preprocessing=True
    )

    # Choose any dataset to make modelling.
    dataset = 'datasets/tanks_256_black.npy'

    # Pass all created to list and run all modellings. TQDM library will show modelling time left.
    cpr_object.work(dataset, [default_config, config])
