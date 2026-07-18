# RelayMetrics.msg
# Performance metrics for the relay system

std_msgs/Header header

# Communication metrics
float64 total_throughput        # Mbps
float64 average_user_rate       # Mbps
float64 min_user_rate           # Mbps (worst user)
float64 max_user_rate           # Mbps (best user)
float64 fairness_index          # Jain's fairness (0-1)
int32 users_served              # Number of users meeting QoS
int32 total_users

# Navigation metrics
float64 trajectory_length       # meters
float64 flight_time             # seconds
float64 energy_consumed         # Joules (estimated)
bool collision_occurred

# System metrics
float64 vla_inference_time      # milliseconds
float64 ego_planning_time       # milliseconds
float64 total_latency           # milliseconds
