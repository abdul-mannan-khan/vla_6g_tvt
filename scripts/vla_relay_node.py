#!/home/it-services/ego_env/bin/python3
"""
VLA Relay Node - Vision-Language-Action Model for 6G UAV Relay Positioning

This node:
1. Receives channel state and UAV camera input
2. Uses a fine-tuned VLA model to determine optimal relay position
3. Publishes target position to EGO Planner

Architecture:
- Input: Channel state (SNR, user positions, throughput) + Optional visual input
- Model: Fine-tuned Llama 3.2 or custom VLA
- Output: Target relay position (x, y, z) for EGO Planner
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Custom messages (will be generated after colcon build)
# from vla_6g_tvt.msg import ChannelState, VLACommand

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json

# For VLA model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re


@dataclass
class UserInfo:
    """Ground user information"""
    position: np.ndarray
    snr: float
    rate: float
    required_rate: float
    is_covered: bool


@dataclass
class ChannelStateData:
    """Processed channel state"""
    bs_position: np.ndarray
    uav_position: np.ndarray
    users: List[UserInfo]
    total_throughput: float
    fairness_index: float


class OptimalRelayCalculator:
    """
    Analytical baseline for computing optimal relay position.
    Used for:
    1. Generating training data for VLA
    2. Comparison baseline
    """

    def __init__(self, frequency_ghz: float = 300.0, bs_power_dbm: float = 30.0,
                 uav_power_dbm: float = 20.0):
        self.frequency_ghz = frequency_ghz
        self.bs_power_dbm = bs_power_dbm
        self.uav_power_dbm = uav_power_dbm

    def compute_optimal_position(self, bs_pos: np.ndarray,
                                  user_positions: List[np.ndarray],
                                  min_altitude: float = 10.0,
                                  max_altitude: float = 40.0) -> Tuple[np.ndarray, str]:
        """
        Compute optimal relay position using geometric analysis.

        Strategy: Position UAV to balance BS-UAV and UAV-Users links
        while maximizing coverage fairness.

        Returns:
            (optimal_position, reasoning_text)
        """
        if not user_positions:
            return np.array([50.0, 50.0, 20.0]), "No users, default position"

        # Compute user centroid (weighted by coverage need)
        user_centroid = np.mean(user_positions, axis=0)

        # Optimal horizontal position: between BS and user centroid
        # Weighted towards users since UAV power is lower
        bs_2d = bs_pos[:2]
        centroid_2d = user_centroid[:2]

        # Weight factor (closer to users due to lower UAV tx power)
        alpha = 0.6  # 60% towards users

        optimal_xy = (1 - alpha) * bs_2d + alpha * centroid_2d

        # Optimal altitude: trade-off between coverage area and path loss
        # Higher = better coverage angle, but more path loss
        # Use geometric mean of distances
        distances_to_users = [np.linalg.norm(optimal_xy - u[:2]) for u in user_positions]
        avg_horizontal_dist = np.mean(distances_to_users)

        # Empirical formula: altitude ~ 0.5 * avg_horizontal_distance, clamped
        optimal_z = np.clip(0.5 * avg_horizontal_dist, min_altitude, max_altitude)

        optimal_pos = np.array([optimal_xy[0], optimal_xy[1], optimal_z])

        # Generate reasoning
        reasoning = (
            f"Positioned between BS and user centroid (alpha={alpha}). "
            f"User centroid: ({centroid_2d[0]:.1f}, {centroid_2d[1]:.1f}). "
            f"Altitude {optimal_z:.1f}m for {len(user_positions)} users with "
            f"avg horizontal distance {avg_horizontal_dist:.1f}m."
        )

        return optimal_pos, reasoning


class VLAModel:
    """
    Vision-Language-Action Model wrapper.

    Phase 1 (Current): Rule-based/analytical decisions
    Phase 2: Fine-tuned Llama model
    Phase 3: Full VLA with vision
    """

    def __init__(self, model_type: str = "analytical", node=None):
        self.model_type = model_type
        self.node = node
        self.model = None
        self.tokenizer = None
        self.calculator = OptimalRelayCalculator()

        if model_type == "llama":
            self._load_llama_model()
        elif model_type == "vla":
            self._load_vla_model()

    def _load_llama_model(self):
        """Load fine-tuned Llama model for relay positioning."""
        model_path = self.node.get_parameter('model_path').get_parameter_value().string_value

        if not model_path:
            model_path = '/home/it-services/ros2_ws/src/vla_6g_tvt/models/vla_6g_v1'

        print(f"[VLA] Loading fine-tuned model from: {model_path}")

        # Base model
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # 4-bit quantization for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()

        print(f"[VLA] Model loaded successfully on {self.model.device}")

    def _load_vla_model(self):
        """Load full VLA model with vision encoder."""
        # Placeholder for VLA model
        print("[VLA] Full VLA model loading - placeholder")

    def predict(self, channel_state: ChannelStateData,
                image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, str]:
        """
        Predict optimal relay position.

        Args:
            channel_state: Current channel state
            image: Optional camera image from UAV

        Returns:
            (target_position, confidence, reasoning)
        """
        start_time = time.time()

        if self.model_type == "analytical":
            position, reasoning = self._analytical_predict(channel_state)
            confidence = 0.9  # High confidence for analytical

        elif self.model_type == "llama":
            position, confidence, reasoning = self._llama_predict(channel_state)

        elif self.model_type == "vla":
            position, confidence, reasoning = self._vla_predict(channel_state, image)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        inference_time = (time.time() - start_time) * 1000  # ms
        reasoning += f" [Inference: {inference_time:.1f}ms]"

        return position, confidence, reasoning

    def _analytical_predict(self, state: ChannelStateData) -> Tuple[np.ndarray, str]:
        """Analytical baseline prediction."""
        user_positions = [u.position for u in state.users]
        return self.calculator.compute_optimal_position(
            state.bs_position, user_positions
        )

    def _llama_predict(self, state: ChannelStateData) -> Tuple[np.ndarray, float, str]:
        """Llama-based prediction using fine-tuned model."""
        # Format input as instruction prompt (matching training format)
        prompt = self._format_llama_prompt(state)

        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=384
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the response part after "### Response:"
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        # Parse position from response
        position, confidence, reasoning = self._parse_llama_response(response, state)

        return position, confidence, reasoning

    def _parse_llama_response(self, response: str, state: ChannelStateData) -> Tuple[np.ndarray, float, str]:
        """Parse Llama model output to extract position."""
        try:
            # 1) Try json.loads on the whole response (or first JSON object found)
            json_match = re.search(r'\{[^{}]+\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if 'x' in data and 'y' in data and 'z' in data:
                        x, y, z = float(data['x']), float(data['y']), float(data['z'])
                        position = np.array([
                            np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)
                        ])
                        return position, 0.90, f"[Llama-json] pos=({x:.1f}, {y:.1f}, {z:.1f})"
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

            # 2) Regex: "x": val, "y": val, "z": val
            simple_pattern = r'"x":\s*([\d.]+)\s*,\s*"y":\s*([\d.]+)\s*,\s*"z":\s*([\d.]+)'
            match = re.search(simple_pattern, response, re.IGNORECASE)
            if match:
                x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                position = np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)])
                return position, 0.85, f"[Llama-regex] pos=({x:.1f}, {y:.1f}, {z:.1f})"

            # 3) Tuple pattern (x, y, z)
            tuple_pattern = r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)'
            match = re.search(tuple_pattern, response)
            if match:
                x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                position = np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)])
                return position, 0.80, f"[Llama-tuple] pos=({x:.1f}, {y:.1f}, {z:.1f})"

            # 4) Last resort: find any three consecutive numbers
            numbers = re.findall(r'(\d+\.?\d*)', response)
            if len(numbers) >= 3:
                x, y, z = float(numbers[0]), float(numbers[1]), float(numbers[2])
                if 0 <= x <= 100 and 0 <= y <= 100 and 1 <= z <= 60:
                    position = np.array([np.clip(x, 0, 100), np.clip(y, 0, 100), np.clip(z, 5, 50)])
                    return position, 0.60, f"[Llama-numbers] pos=({x:.1f}, {y:.1f}, {z:.1f})"

            # 5) Fallback to analytical
            position, analytical_reasoning = self._analytical_predict(state)
            return position, 0.5, f"[Llama-fallback] {analytical_reasoning}"

        except Exception as e:
            position, analytical_reasoning = self._analytical_predict(state)
            return position, 0.5, f"[Llama-error] {str(e)[:50]}. Fallback: {analytical_reasoning}"

    def _vla_predict(self, state: ChannelStateData,
                     image: Optional[np.ndarray]) -> Tuple[np.ndarray, float, str]:
        """Full VLA prediction with vision (placeholder)."""
        pos, reasoning = self._analytical_predict(state)
        return pos, 0.8, f"[VLA-placeholder] {reasoning}"

    def _format_llama_prompt(self, state: ChannelStateData) -> str:
        """Format channel state as Llama prompt."""
        prompt = f"""You are a UAV relay positioning expert for 6G networks.

Current situation:
- Base station at: ({state.bs_position[0]:.1f}, {state.bs_position[1]:.1f}, {state.bs_position[2]:.1f})
- UAV currently at: ({state.uav_position[0]:.1f}, {state.uav_position[1]:.1f}, {state.uav_position[2]:.1f})
- Number of ground users: {len(state.users)}
- Current total throughput: {state.total_throughput:.1f} Mbps
- Current fairness index: {state.fairness_index:.3f}

User details:
"""
        for i, user in enumerate(state.users):
            prompt += f"  User {i}: pos=({user.position[0]:.1f}, {user.position[1]:.1f}), "
            prompt += f"SNR={user.snr:.1f}dB, rate={user.rate:.1f}Mbps, "
            prompt += f"required={user.required_rate:.1f}Mbps, covered={user.is_covered}\n"

        prompt += """
Task: Determine the optimal UAV relay position.
Output ONLY valid JSON: {"x": float, "y": float, "z": float}
"""
        return prompt


class VLARelayNode(Node):
    """ROS2 Node for VLA-based relay positioning."""

    def __init__(self):
        super().__init__('vla_relay_node')

        # Parameters
        self.declare_parameter('model_type', 'analytical')  # analytical, llama, vla
        self.declare_parameter('model_path', '')            # Path to fine-tuned model
        self.declare_parameter('update_rate_hz', 2.0)       # VLA decision rate
        self.declare_parameter('min_position_change', 2.0)  # meters
        self.declare_parameter('confidence_threshold', 0.5)

        model_type = self.get_parameter('model_type').value
        self.update_rate = self.get_parameter('update_rate_hz').value
        self.min_position_change = self.get_parameter('min_position_change').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value

        # Initialize VLA model
        self.vla_model = VLAModel(model_type=model_type, node=self)

        # State variables
        self.current_channel_state = None
        self.current_uav_position = np.array([25.0, 25.0, 20.0])
        self.last_target_position = None
        self.sequence_id = 0

        # Base station position (should match channel simulator)
        self.bs_position = np.array([0.0, 0.0, 30.0])

        # Publishers
        # Publish target to EGO Planner via waypoint topic
        self.waypoint_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',  # EGO Planner listens to this
            10
        )

        # Also publish structured VLA command for logging/analysis
        # self.vla_command_pub = self.create_publisher(VLACommand, 'vla_command', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom_world',
            self.odom_callback,
            10
        )

        # Channel state subscriber (using raw messages for now)
        # Will use custom message after build
        # self.channel_sub = self.create_subscription(
        #     ChannelState, 'channel_state', self.channel_callback, 10)

        # Timer for VLA decisions
        self.timer = self.create_timer(
            1.0 / self.update_rate,
            self.vla_decision_callback
        )

        self.get_logger().info(f'VLA Relay Node initialized with model_type={model_type}')
        self.get_logger().info(f'Decision rate: {self.update_rate} Hz')

    def odom_callback(self, msg: Odometry):
        """Update current UAV position."""
        self.current_uav_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

    def channel_callback(self, msg):
        """Process channel state update."""
        # Parse message into internal format
        users = []
        for i in range(len(msg.user_positions)):
            user = UserInfo(
                position=np.array([
                    msg.user_positions[i].x,
                    msg.user_positions[i].y,
                    msg.user_positions[i].z
                ]),
                snr=msg.snr_uav_user[i] if i < len(msg.snr_uav_user) else 0.0,
                rate=msg.user_rates[i] if i < len(msg.user_rates) else 0.0,
                required_rate=100.0,  # Default, should come from user state
                is_covered=bool(msg.coverage_status[i]) if i < len(msg.coverage_status) else False
            )
            users.append(user)

        self.current_channel_state = ChannelStateData(
            bs_position=np.array([msg.bs_position.x, msg.bs_position.y, msg.bs_position.z]),
            uav_position=self.current_uav_position,
            users=users,
            total_throughput=msg.total_throughput,
            fairness_index=msg.fairness_index
        )

    def vla_decision_callback(self):
        """Main VLA decision loop."""
        # Create synthetic channel state if not receiving real data
        if self.current_channel_state is None:
            self.current_channel_state = self._create_synthetic_state()

        # Run VLA prediction
        target_pos, confidence, reasoning = self.vla_model.predict(
            self.current_channel_state
        )

        # Check if we should publish new target
        should_publish = False

        if self.last_target_position is None:
            should_publish = True
        else:
            distance = np.linalg.norm(target_pos - self.last_target_position)
            if distance > self.min_position_change:
                should_publish = True

        if confidence < self.confidence_threshold:
            should_publish = False
            self.get_logger().warn(f'Low confidence ({confidence:.2f}), skipping update')

        if should_publish:
            self._publish_target(target_pos, confidence, reasoning)
            self.last_target_position = target_pos.copy()

    def _create_synthetic_state(self) -> ChannelStateData:
        """Create synthetic channel state for testing."""
        # Default user positions
        user_positions = [
            np.array([40.0, 40.0, 1.0]),
            np.array([60.0, 30.0, 1.0]),
            np.array([50.0, 60.0, 1.0]),
            np.array([30.0, 50.0, 1.0]),
            np.array([70.0, 50.0, 1.0]),
        ]

        users = []
        for i, pos in enumerate(user_positions):
            users.append(UserInfo(
                position=pos,
                snr=20.0 - i * 2,  # Decreasing SNR
                rate=500.0 - i * 50,
                required_rate=100.0,
                is_covered=True
            ))

        return ChannelStateData(
            bs_position=self.bs_position,
            uav_position=self.current_uav_position,
            users=users,
            total_throughput=sum(u.rate for u in users),
            fairness_index=0.85
        )

    def _publish_target(self, position: np.ndarray, confidence: float, reasoning: str):
        """Publish target position to EGO Planner."""
        # PoseStamped for EGO Planner waypoint
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.w = 1.0

        self.waypoint_pub.publish(msg)
        self.sequence_id += 1

        self.get_logger().info(
            f'VLA target #{self.sequence_id}: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}) '
            f'conf={confidence:.2f}'
        )
        self.get_logger().info(f'  Reasoning: {reasoning}')


def main(args=None):
    rclpy.init(args=args)
    node = VLARelayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
