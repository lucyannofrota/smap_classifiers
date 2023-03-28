

# import launch
import launch
from launch import LaunchDescription
import launch_ros.actions
from launch.actions import (IncludeLaunchDescription)
from launch.actions import DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_context import LaunchContext
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration

# import os
import yaml
#from yaml import load
from yaml import CLoader as Loader
#from smap_classification_wrapper.classification_wrapper import main
#from smap_perception_yolo_v5.yolo_v5_node import yolo_v5
import launch
import launch_ros.actions


def generate_launch_description():

    model_launch_arg = DeclareLaunchArgument(
        "model", default_value=TextSubstitution(
        text=PathJoinSubstitution([
            FindPackageShare('smap_perception_yolo_v5'),
            'weights/yolov5s.torchscript'
            ]).perform(LaunchContext())
        ),
        description='Model path [*.torchscript]'
    )

    model_description_launch_arg = DeclareLaunchArgument(
        "model_description", default_value=TextSubstitution(
        text=PathJoinSubstitution([
            FindPackageShare('smap_perception_yolo_v5'),
            'data/coco128.yaml'
            ]).perform(LaunchContext())
        ),
        description='Model description path [*.yaml]'
    )

    yolo_v5_node = launch_ros.actions.Node(
            package='smap_perception_yolo_v5',
            executable='yolo_v5_node.py',
            parameters=[{
                "model": LaunchConfiguration('model'),
                "model_description": LaunchConfiguration('model_description')
            }],
            output='screen',
            arguments=[('__log_level:=debug')],
            emulate_tty=True
    )

    return LaunchDescription([
        model_launch_arg,
        model_description_launch_arg,
        yolo_v5_node
    ])

if __name__ == '__main__':
    generate_launch_description()
