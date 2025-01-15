from setuptools import find_packages, setup

package_name = 'MBK_env'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'std_msgs'],
    zip_safe=True,
    maintainer='ali',
    maintainer_email='alibaniasad1999@yahoo.com',
    description='A package that publishes and subscribes to an array for MBK system.',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'MBK_env_publisher_node = MBK_env.MBK_env_publisher:main',
            'MBK_env_subscriber_node = MBK_env.mbk_plotter:main',
        ],
    },
)
