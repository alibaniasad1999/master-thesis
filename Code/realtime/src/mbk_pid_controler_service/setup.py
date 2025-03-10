from setuptools import find_packages, setup

package_name = 'mbk_pid_controler_service'

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
                    'controller_service = mbk_pid_controler_service.mbk_controler:main',
                    'model_service = mbk_pid_controler_service.mbk_model:main',
                ],
    },
)
