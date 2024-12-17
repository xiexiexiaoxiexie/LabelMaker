import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_sample_pose():
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]  # Some translation
    return pose

def complex_transform(pose_mat):
    pose_mat = pose_mat.copy()
    
    # Part 1: Mirror coordinates
    rotation = pose_mat[:3, :3]
    rotation[2, :] = -rotation[2, :]
    rotation[1, :] = -rotation[1, :]
    rotation[0, :] = -rotation[0, :]
    pose_mat[:3, :3] = rotation
    pose_mat[:, 0] = -pose_mat[:, 0]
    
    # Part 2: Apply rotations
    # R_x_90 = np.array([[1, 0, 0, 0],
    #     [0, np.cos(np.radians(90)), -np.sin(np.radians(90)), 0],
    #     [0, np.sin(np.radians(90)), np.cos(np.radians(90)), 0],
    #     [0, 0, 0, 1]])
    
    # R_z_90 = np.array([[np.cos(np.radians(-90)), -np.sin(np.radians(-90)), 0, 0],
    #     [np.sin(np.radians(-90)), np.cos(np.radians(-90)), 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]])
    
    # R_z_x = np.dot(R_z_90, R_x_90)
    # return np.dot(R_z_x, pose_mat)
    # return pose_mat

def simple_transform(pose_mat):
    result = pose_mat.copy()
    # Just swap Y and Z while negating Z
    result[:3, :] = pose_mat[[0, 2, 1], :]  # Reorder rows
    result[2, :] = -result[2, :]  # Negate Z
    return result

def plot_coordinate_frame(ax, pose_mat, colors, label_prefix=""):
    """Plot coordinate frame arrows starting from origin."""
    origin = pose_mat[:3, 3]   # Always start from (0,0,0)
    
    axis_names = ['X', 'Y', 'Z']
    scale = 1.0  # Increased scale for better visibility
    
    arrows = []
    for i in range(3):
        direction = pose_mat[:3, i]
        arrow = ax.quiver(origin[0], origin[1], origin[2],
                       direction[0] * scale, direction[1] * scale, direction[2] * scale,
                       color=colors[i], label=f'{label_prefix}{axis_names[i]}',
                       linewidth=2)
        arrows.append(arrow)
    return arrows

# Create sample pose
original_pose = create_sample_pose()
complex_result = complex_transform(original_pose)
simple_result = simple_transform(original_pose)

# Create figure
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot all three coordinate frames
plot_coordinate_frame(ax, original_pose, ['crimson', 'forestgreen', 'royalblue'], "Original ")
plot_coordinate_frame(ax, complex_result, ['darkred', 'darkgreen', 'navy'], "Complex ")
# plot_coordinate_frame(ax, simple_result, ['orangered', 'limegreen', 'dodgerblue'], "Simple ")

# Set consistent viewing angles and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=45)

# Add title and legend
ax.set_title('Coordinate Transformations at Origin')
ax.legend(bbox_to_anchor=(1.15, 1))

# Make the plot more visually appealing
# ax.set_box_aspect([1,1,1])
ax.grid(True)

# Set equal axis limits centered on origin
limit = 5.2
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

# Add origin point
ax.scatter([0], [0], [0], color='black', s=100, label='Origin')

# Add axes planes for better orientation
xx, yy = np.meshgrid([-limit, limit], [-limit, limit])
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')  # XY plane
ax.plot_surface(xx, zz, yy, alpha=0.1, color='gray')  # XZ plane
ax.plot_surface(zz, xx, yy, alpha=0.1, color='gray')  # YZ plane
plt.show()
# Print verification of equivalence
print("Are transforms equivalent?", 
      np.allclose(complex_result, simple_result))

# Print the axes directions in a more readable format
def print_axes(pose, name):
    print(f"\n{name} axes directions:")
    print(f"X: {pose[:3, 0]}")
    print(f"Y: {pose[:3, 1]}")
    print(f"Z: {pose[:3, 2]}")

print_axes(original_pose, "Original")
print_axes(complex_result, "Complex transform")
print_axes(simple_result, "Simple transform")