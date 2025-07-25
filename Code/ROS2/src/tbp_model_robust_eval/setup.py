from setuptools import find_packages, setup

package_name = 'tbp_model_robust_eval'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ali',
    maintainer_email='alibaniasad1999@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tbp_model = tbp_model_robust_eval.TBP_model:main',
            'actuator_disturbance = tbp_model_robust_eval.actuator_disturbance:main',
            'initial_condition_shift = tbp_model_robust_eval.initial_condition_shift:main',
            'model_mismatch = tbp_model_robust_eval.model_mismatch:main',
            'partial_observation = tbp_model_robust_eval.partial_observation:main',
            'sensor_noise = tbp_model_robust_eval.sensor_noise:main',
            'time_delay = tbp_model_robust_eval.time_delay:main',
        ],
    },
)
