/**
 * @file users_simulator_node.cpp
 * @brief Ground Users Simulator for 6G UAV Relay System
 *
 * Simulates ground users with configurable:
 * - Static or moving behavior
 * - QoS requirements
 * - Positions
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "vla_6g_relay/msg/user_state.hpp"

#include <cmath>
#include <vector>
#include <random>

using namespace std::chrono_literals;

struct GroundUser
{
    int id;
    double x, y, z;
    double vx, vy;           // Velocity (for moving users)
    double required_rate;    // QoS requirement (Mbps)
    bool is_moving;
};

class UsersSimulator : public rclcpp::Node
{
public:
    UsersSimulator() : Node("users_simulator")
    {
        // Parameters
        this->declare_parameter("num_users", 5);
        this->declare_parameter("area_min_x", 20.0);
        this->declare_parameter("area_max_x", 80.0);
        this->declare_parameter("area_min_y", 20.0);
        this->declare_parameter("area_max_y", 80.0);
        this->declare_parameter("moving_users_fraction", 0.0);  // 0 = all static
        this->declare_parameter("max_user_speed", 2.0);         // m/s
        this->declare_parameter("default_required_rate", 100.0); // Mbps
        this->declare_parameter("update_rate_hz", 10.0);

        num_users_ = this->get_parameter("num_users").as_int();
        area_min_x_ = this->get_parameter("area_min_x").as_double();
        area_max_x_ = this->get_parameter("area_max_x").as_double();
        area_min_y_ = this->get_parameter("area_min_y").as_double();
        area_max_y_ = this->get_parameter("area_max_y").as_double();
        moving_fraction_ = this->get_parameter("moving_users_fraction").as_double();
        max_speed_ = this->get_parameter("max_user_speed").as_double();
        default_rate_ = this->get_parameter("default_required_rate").as_double();

        // Initialize random generator
        std::random_device rd;
        gen_ = std::mt19937(rd());

        // Initialize users
        initializeUsers();

        // Publishers
        user_state_pub_ = this->create_publisher<vla_6g_relay::msg::UserState>(
            "user_states", 10);

        visualization_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "users_visualization", 10);

        // Timer
        double rate = this->get_parameter("update_rate_hz").as_double();
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / rate)),
            std::bind(&UsersSimulator::updateUsers, this));

        RCLCPP_INFO(this->get_logger(), "Users Simulator initialized with %d users", num_users_);
    }

private:
    void initializeUsers()
    {
        std::uniform_real_distribution<> x_dist(area_min_x_, area_max_x_);
        std::uniform_real_distribution<> y_dist(area_min_y_, area_max_y_);
        std::uniform_real_distribution<> speed_dist(-max_speed_, max_speed_);
        std::uniform_real_distribution<> rate_dist(50.0, 200.0);  // 50-200 Mbps

        int num_moving = static_cast<int>(num_users_ * moving_fraction_);

        for (int i = 0; i < num_users_; ++i)
        {
            GroundUser user;
            user.id = i;
            user.x = x_dist(gen_);
            user.y = y_dist(gen_);
            user.z = 1.0;  // Ground level
            user.is_moving = (i < num_moving);

            if (user.is_moving)
            {
                user.vx = speed_dist(gen_);
                user.vy = speed_dist(gen_);
            }
            else
            {
                user.vx = 0.0;
                user.vy = 0.0;
            }

            user.required_rate = rate_dist(gen_);
            users_.push_back(user);

            RCLCPP_INFO(this->get_logger(),
                "User %d: pos=(%.1f, %.1f), moving=%d, required_rate=%.1f Mbps",
                user.id, user.x, user.y, user.is_moving, user.required_rate);
        }
    }

    void updateUsers()
    {
        double dt = 0.1;  // 10 Hz update

        for (auto& user : users_)
        {
            // Update position for moving users
            if (user.is_moving)
            {
                user.x += user.vx * dt;
                user.y += user.vy * dt;

                // Bounce off boundaries
                if (user.x < area_min_x_ || user.x > area_max_x_)
                {
                    user.vx = -user.vx;
                    user.x = std::clamp(user.x, area_min_x_, area_max_x_);
                }
                if (user.y < area_min_y_ || user.y > area_max_y_)
                {
                    user.vy = -user.vy;
                    user.y = std::clamp(user.y, area_min_y_, area_max_y_);
                }
            }

            // Publish user state
            auto msg = vla_6g_relay::msg::UserState();
            msg.header.stamp = this->now();
            msg.header.frame_id = "world";
            msg.user_id = user.id;
            msg.position.x = user.x;
            msg.position.y = user.y;
            msg.position.z = user.z;
            msg.velocity.x = user.vx;
            msg.velocity.y = user.vy;
            msg.velocity.z = 0.0;
            msg.required_rate = user.required_rate;
            msg.current_rate = 0.0;  // Will be filled by channel simulator
            msg.is_served = false;

            user_state_pub_->publish(msg);
        }

        // Publish visualization
        publishVisualization();
    }

    void publishVisualization()
    {
        visualization_msgs::msg::MarkerArray markers;

        for (const auto& user : users_)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = this->now();
            marker.ns = "ground_users";
            marker.id = user.id;
            marker.type = visualization_msgs::msg::Marker::CYLINDER;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = user.x;
            marker.pose.position.y = user.y;
            marker.pose.position.z = user.z;

            marker.scale.x = 2.0;
            marker.scale.y = 2.0;
            marker.scale.z = 1.8;

            // Color based on movement: green=static, orange=moving
            if (user.is_moving)
            {
                marker.color.r = 1.0;
                marker.color.g = 0.5;
                marker.color.b = 0.0;
            }
            else
            {
                marker.color.r = 0.0;
                marker.color.g = 0.8;
                marker.color.b = 0.0;
            }
            marker.color.a = 0.8;

            markers.markers.push_back(marker);

            // Add text label with user ID
            visualization_msgs::msg::Marker text_marker;
            text_marker.header.frame_id = "world";
            text_marker.header.stamp = this->now();
            text_marker.ns = "user_labels";
            text_marker.id = user.id;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;

            text_marker.pose.position.x = user.x;
            text_marker.pose.position.y = user.y;
            text_marker.pose.position.z = user.z + 3.0;

            text_marker.scale.z = 1.5;
            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;

            text_marker.text = "U" + std::to_string(user.id);
            markers.markers.push_back(text_marker);
        }

        visualization_pub_->publish(markers);
    }

    // Parameters
    int num_users_;
    double area_min_x_, area_max_x_;
    double area_min_y_, area_max_y_;
    double moving_fraction_;
    double max_speed_;
    double default_rate_;

    // State
    std::vector<GroundUser> users_;
    std::mt19937 gen_;

    // ROS interfaces
    rclcpp::Publisher<vla_6g_relay::msg::UserState>::SharedPtr user_state_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualization_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UsersSimulator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
