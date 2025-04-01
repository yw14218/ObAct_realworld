import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf_transformations  # ensure this is installed (often via pip install tf-transformations)

# Example transform list entries: each entry is [x, y, z, roll, pitch, yaw]
TRANSFORMS = [
    [0.33984852, 0.02906636, 0.05744444, -0.26785669, 1.09211407, 0.01706309],
    [0.34046315, 0.02783402, 0.0568135, -0.25664683, 1.09079213, 0.02944555],
    [0.3416057, 0.02761643, 0.05623637, -0.2490271, 1.08940631, 0.03535426],
    [0.34233899, 0.02706255, 0.0556202, -0.24244448, 1.08820495, 0.04220409],
    [0.34307416, 0.02708848, 0.05487901, -0.23671955, 1.08889758, 0.04809022],
    [0.34375268, 0.02730636, 0.05493213, -0.23345261, 1.0899616, 0.05428814],
    [0.34460585, 0.02751212, 0.05336623, -0.22626726, 1.0871515, 0.06533454],
    [0.34529858, 0.02766355, 0.05286645, -0.22454029, 1.08784116, 0.07063049],
    [0.34588135, 0.02683915, 0.05227532, -0.22251839, 1.08810682, 0.07794519],
    [0.34657588, 0.02775786, 0.05158724, -0.21955996, 1.08751935, 0.0845217]
]

class TransformPublisher(Node):
    def __init__(self):
        super().__init__('transform_publisher')
        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        self.transforms = TRANSFORMS

        # Publish all transforms every second.
        self.timer = self.create_timer(1.0, self.publish_all_transforms)

    def publish_all_transforms(self):
        now = self.get_clock().now().to_msg()
        for i, trans in enumerate(self.transforms):
            x, y, z, roll, pitch, yaw = trans

            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'arm_1/base_link'
            # Use the index in the child frame name.
            t.child_frame_id = f'arm_1/end_effector_{i}'

            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z

            quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            self.broadcaster.sendTransform(t)
            self.get_logger().info(f"Published transform {i}: {trans}")

def main(args=None):
    rclpy.init(args=args)
    node = TransformPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # allow ctrl-c to stop the node
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

