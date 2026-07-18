# ChannelState.msg
# Represents the current 6G channel state for the UAV relay system

std_msgs/Header header

# Base station position
geometry_msgs/Point bs_position

# UAV current position
geometry_msgs/Point uav_position

# Ground users positions (array)
geometry_msgs/Point[] user_positions

# Signal-to-Noise Ratio for each BS-UAV-User link (dB)
float64[] snr_bs_uav      # BS to UAV link SNR
float64[] snr_uav_user    # UAV to each user SNR

# Achievable data rate for each user (Mbps)
float64[] user_rates

# Total system throughput (Mbps)
float64 total_throughput

# Coverage indicator (1 if user is covered, 0 otherwise)
int32[] coverage_status

# Fairness index (Jain's fairness)
float64 fairness_index
