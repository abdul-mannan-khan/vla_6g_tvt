# VLACommand.msg
# Command from VLA model to EGO Planner

std_msgs/Header header

# Target relay position determined by VLA
geometry_msgs/Point target_position

# Confidence score (0-1) of the VLA decision
float64 confidence

# Priority level (higher = more urgent repositioning)
int32 priority

# Reasoning text (for explainability/debugging)
string reasoning

# Whether this is a new target or update
bool is_new_target

# Sequence number for tracking
uint32 sequence_id
