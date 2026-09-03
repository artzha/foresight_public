from glob import glob
import os

from setuptools import find_packages, setup

package_name = "legged_deployment"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [os.path.join("resource", package_name)],
        ),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "cotnav", "pyyaml", "scipy"],
    zip_safe=True,
    maintainer="Arthur Zhang",
    maintainer_email="arthurz@cs.utexas.edu",
    description="ROS2 deployment package for cotnav vLLM serving and path inference.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "waypoint_planner = legged_deployment.waypoint_planner_node:main",
            "lelan_planner = legged_deployment.lelan_planner_node:main",
        ],
    },
)
