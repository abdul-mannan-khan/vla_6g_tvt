/**
 * @file channel_simulator_node.cpp
 * @brief Simplified 6G THz Channel Simulator for UAV Relay
 *
 * This node simulates the wireless channel between:
 * - Base Station (BS) and UAV
 * - UAV and Ground Users (GUs)
 *
 * Channel Model: Free-space path loss + simplified THz absorption
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "vla_6g_relay/msg/channel_state.hpp"
#include "vla_6g_relay/msg/user_state.hpp"

#include <cmath>
#include <vector>
#include <random>

using namespace std::chrono_literals;

class ChannelSimulator : public rclcpp::Node
{
public:
    ChannelSimulator() : Node("channel_simulator")
    {
        // Declare parameters
        this->declare_parameter("frequency_ghz", 300.0);        // 300 GHz (THz band)
        this->declare_parameter("bandwidth_ghz", 10.0);         // 10 GHz bandwidth
        this->declare_parameter("bs_tx_power_dbm", 30.0);       // BS transmit power
        this->declare_parameter("uav_tx_power_dbm", 20.0);      // UAV relay power
        this->declare_parameter("noise_figure_db", 10.0);       // Receiver noise figure
        this->declare_parameter("update_rate_hz", 10.0);        // Channel update rate

        // Base station position
        this->declare_parameter("bs_position_x", 0.0);
        this->declare_parameter("bs_position_y", 0.0);
        this->declare_parameter("bs_position_z", 30.0);         // BS on building/tower

        // Get parameters
        frequency_ghz_ = this->get_parameter("frequency_ghz").as_double();
        bandwidth_ghz_ = this->get_parameter("bandwidth_ghz").as_double();
        bs_tx_power_dbm_ = this->get_parameter("bs_tx_power_dbm").as_double();
        uav_tx_power_dbm_ = this->get_parameter("uav_tx_power_dbm").as_double();
        noise_figure_db_ = this->get_parameter("noise_figure_db").as_double();

        bs_position_.x = this->get_parameter("bs_position_x").as_double();
        bs_position_.y = this->get_parameter("bs_position_y").as_double();
        bs_position_.z = this->get_parameter("bs_position_z").as_double();

        // Calculate noise power
        // N = kTB + NF, where k=1.38e-23, T=290K, B in Hz
        double bandwidth_hz = bandwidth_ghz_ * 1e9;
        noise_power_dbm_ = 10 * std::log10(1.38e-23 * 290 * bandwidth_hz * 1000) + noise_figure_db_;

        // Initialize UAV position (will be updated from odometry)
        uav_position_.x = 25.0;
        uav_position_.y = 25.0;
        uav_position_.z = 20.0;

        // Publishers
        channel_state_pub_ = this->create_publisher<vla_6g_relay::msg::ChannelState>(
            "channel_state", 10);

        visualization_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "channel_visualization", 10);

        // Subscribers
        uav_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom_world", 10,
            std::bind(&ChannelSimulator::uavOdomCallback, this, std::placeholders::_1));

        users_sub_ = this->create_subscription<vla_6g_relay::msg::UserState>(
            "user_states", 10,
            std::bind(&ChannelSimulator::userStateCallback, this, std::placeholders::_1));

        // Timer for periodic channel updates
        double update_rate = this->get_parameter("update_rate_hz").as_double();
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / update_rate)),
            std::bind(&ChannelSimulator::updateChannel, this));

        RCLCPP_INFO(this->get_logger(), "Channel Simulator initialized");
        RCLCPP_INFO(this->get_logger(), "  Frequency: %.1f GHz, Bandwidth: %.1f GHz",
                    frequency_ghz_, bandwidth_ghz_);
        RCLCPP_INFO(this->get_logger(), "  BS Position: (%.1f, %.1f, %.1f)",
                    bs_position_.x, bs_position_.y, bs_position_.z);
    }

private:
    void uavOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        uav_position_.x = msg->pose.pose.position.x;
        uav_position_.y = msg->pose.pose.position.y;
        uav_position_.z = msg->pose.pose.position.z;
        have_uav_odom_ = true;
    }

    void userStateCallback(const vla_6g_relay::msg::UserState::SharedPtr msg)
    {
        // Update user position in our local storage
        int user_id = msg->user_id;
        if (user_id >= 0 && user_id < static_cast<int>(user_positions_.size()))
        {
            user_positions_[user_id] = msg->position;
        }
        else if (user_id == static_cast<int>(user_positions_.size()))
        {
            user_positions_.push_back(msg->position);
            user_required_rates_.push_back(msg->required_rate);
        }
    }

    void updateChannel()
    {
        if (user_positions_.empty())
        {
            // Initialize with default users if none received
            initializeDefaultUsers();
        }

        auto channel_msg = vla_6g_relay::msg::ChannelState();
        channel_msg.header.stamp = this->now();
        channel_msg.header.frame_id = "world";

        channel_msg.bs_position = bs_position_;
        channel_msg.uav_position = uav_position_;
        channel_msg.user_positions = user_positions_;

        // Calculate channel for each user
        double total_throughput = 0.0;
        int users_covered = 0;
        std::vector<double> user_rates;

        for (size_t i = 0; i < user_positions_.size(); ++i)
        {
            // BS to UAV link
            double d_bs_uav = calculateDistance(bs_position_, uav_position_);
            double snr_bs_uav = calculateSNR(d_bs_uav, bs_tx_power_dbm_);

            // UAV to User link
            double d_uav_user = calculateDistance(uav_position_, user_positions_[i]);
            double snr_uav_user = calculateSNR(d_uav_user, uav_tx_power_dbm_);

            // Relay capacity is limited by the weaker link
            double effective_snr = std::min(snr_bs_uav, snr_uav_user);

            // Shannon capacity (simplified): C = B * log2(1 + SNR)
            double snr_linear = std::pow(10.0, effective_snr / 10.0);
            double rate_mbps = bandwidth_ghz_ * 1000 * std::log2(1.0 + snr_linear) / 1e6;

            // Cap at reasonable maximum
            rate_mbps = std::min(rate_mbps, 10000.0);  // 10 Gbps max

            channel_msg.snr_bs_uav.push_back(snr_bs_uav);
            channel_msg.snr_uav_user.push_back(snr_uav_user);
            channel_msg.user_rates.push_back(rate_mbps);

            // Check if user is covered (meets required rate)
            double required_rate = (i < user_required_rates_.size()) ?
                                   user_required_rates_[i] : 100.0;  // Default 100 Mbps

            int covered = (rate_mbps >= required_rate) ? 1 : 0;
            channel_msg.coverage_status.push_back(covered);

            if (covered) users_covered++;
            total_throughput += rate_mbps;
            user_rates.push_back(rate_mbps);
        }

        channel_msg.total_throughput = total_throughput;

        // Calculate Jain's fairness index
        if (!user_rates.empty())
        {
            double sum = 0.0, sum_sq = 0.0;
            for (double rate : user_rates)
            {
                sum += rate;
                sum_sq += rate * rate;
            }
            double n = static_cast<double>(user_rates.size());
            channel_msg.fairness_index = (sum * sum) / (n * sum_sq);
        }
        else
        {
            channel_msg.fairness_index = 0.0;
        }

        channel_state_pub_->publish(channel_msg);

        // Publish visualization
        publishVisualization();
    }

    double calculateDistance(const geometry_msgs::msg::Point& p1,
                             const geometry_msgs::msg::Point& p2)
    {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        double dz = p1.z - p2.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    double calculateSNR(double distance_m, double tx_power_dbm)
    {
        // Free-space path loss: FSPL = 20*log10(d) + 20*log10(f) + 92.45
        // where d in km, f in GHz
        double d_km = distance_m / 1000.0;
        double fspl_db = 20 * std::log10(d_km) + 20 * std::log10(frequency_ghz_) + 92.45;

        // THz atmospheric absorption (simplified): ~10 dB/km at 300 GHz
        double absorption_db = 10.0 * d_km;

        // Total path loss
        double path_loss_db = fspl_db + absorption_db;

        // Received power
        double rx_power_dbm = tx_power_dbm - path_loss_db;

        // SNR
        double snr_db = rx_power_dbm - noise_power_dbm_;

        return snr_db;
    }

    void initializeDefaultUsers()
    {
        // Initialize 5 default ground users
        std::vector<std::pair<double, double>> default_positions = {
            {40.0, 40.0},
            {60.0, 30.0},
            {50.0, 60.0},
            {30.0, 50.0},
            {70.0, 50.0}
        };

        for (const auto& pos : default_positions)
        {
            geometry_msgs::msg::Point p;
            p.x = pos.first;
            p.y = pos.second;
            p.z = 1.0;  // Ground level
            user_positions_.push_back(p);
            user_required_rates_.push_back(100.0);  // 100 Mbps default requirement
        }

        RCLCPP_INFO(this->get_logger(), "Initialized %zu default ground users",
                    user_positions_.size());
    }

    void publishVisualization()
    {
        visualization_msgs::msg::MarkerArray markers;

        // BS marker
        visualization_msgs::msg::Marker bs_marker;
        bs_marker.header.frame_id = "world";
        bs_marker.header.stamp = this->now();
        bs_marker.ns = "base_station";
        bs_marker.id = 0;
        bs_marker.type = visualization_msgs::msg::Marker::CUBE;
        bs_marker.action = visualization_msgs::msg::Marker::ADD;
        bs_marker.pose.position = bs_position_;
        bs_marker.scale.x = 2.0;
        bs_marker.scale.y = 2.0;
        bs_marker.scale.z = 5.0;
        bs_marker.color.r = 0.0;
        bs_marker.color.g = 0.0;
        bs_marker.color.b = 1.0;
        bs_marker.color.a = 1.0;
        markers.markers.push_back(bs_marker);

        // User markers
        for (size_t i = 0; i < user_positions_.size(); ++i)
        {
            visualization_msgs::msg::Marker user_marker;
            user_marker.header.frame_id = "world";
            user_marker.header.stamp = this->now();
            user_marker.ns = "users";
            user_marker.id = static_cast<int>(i);
            user_marker.type = visualization_msgs::msg::Marker::CYLINDER;
            user_marker.action = visualization_msgs::msg::Marker::ADD;
            user_marker.pose.position = user_positions_[i];
            user_marker.scale.x = 1.5;
            user_marker.scale.y = 1.5;
            user_marker.scale.z = 2.0;
            user_marker.color.r = 0.0;
            user_marker.color.g = 1.0;
            user_marker.color.b = 0.0;
            user_marker.color.a = 1.0;
            markers.markers.push_back(user_marker);
        }

        // Link lines (BS-UAV-Users)
        visualization_msgs::msg::Marker link_marker;
        link_marker.header.frame_id = "world";
        link_marker.header.stamp = this->now();
        link_marker.ns = "links";
        link_marker.id = 0;
        link_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        link_marker.action = visualization_msgs::msg::Marker::ADD;
        link_marker.scale.x = 0.1;
        link_marker.color.r = 1.0;
        link_marker.color.g = 1.0;
        link_marker.color.b = 0.0;
        link_marker.color.a = 0.5;

        // BS to UAV link
        link_marker.points.push_back(bs_position_);
        link_marker.points.push_back(uav_position_);

        // UAV to each user
        for (const auto& user_pos : user_positions_)
        {
            link_marker.points.push_back(uav_position_);
            link_marker.points.push_back(user_pos);
        }
        markers.markers.push_back(link_marker);

        visualization_pub_->publish(markers);
    }

    // Parameters
    double frequency_ghz_;
    double bandwidth_ghz_;
    double bs_tx_power_dbm_;
    double uav_tx_power_dbm_;
    double noise_figure_db_;
    double noise_power_dbm_;

    // State
    geometry_msgs::msg::Point bs_position_;
    geometry_msgs::msg::Point uav_position_;
    std::vector<geometry_msgs::msg::Point> user_positions_;
    std::vector<double> user_required_rates_;
    bool have_uav_odom_ = false;

    // ROS interfaces
    rclcpp::Publisher<vla_6g_relay::msg::ChannelState>::SharedPtr channel_state_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr visualization_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr uav_odom_sub_;
    rclcpp::Subscription<vla_6g_relay::msg::UserState>::SharedPtr users_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ChannelSimulator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
