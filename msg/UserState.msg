# UserState.msg
# State of a single ground user

std_msgs/Header header

# User ID
int32 user_id

# Current position
geometry_msgs/Point position

# Velocity (for moving users, optional)
geometry_msgs/Vector3 velocity

# Required data rate (QoS requirement in Mbps)
float64 required_rate

# Current achieved rate (Mbps)
float64 current_rate

# Is user currently served (has acceptable QoS)
bool is_served
