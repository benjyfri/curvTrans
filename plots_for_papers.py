from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def generate_random_points():
    matrix = np.random.uniform(1, 6, (3, 3))
    return matrix


def calculate_center_of_mass(points):
    return np.mean(points, axis=0)


def create_point_cloud(points, M, o):
    return np.vstack([o, points, M])


def rotate_point_cloud_to_align(M_point, point_cloud, epsilon=1e-8):
    M_norm = np.linalg.norm(M_point)
    v1 = M_point / (M_norm + epsilon)
    v2 = np.array([0.0, 0.0, 1.0])

    rotation_axis = np.cross(v1, v2)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    rotation_axis = rotation_axis / (rotation_axis_norm + epsilon)

    cos_theta = np.dot(v1, v2)
    sin_theta = np.sqrt(1 - cos_theta ** 2 + epsilon)

    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])

    R1 = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
    rotated_points = np.dot(point_cloud, R1.T)

    return rotated_points


def rotate_point_z_to_x_axis(p, pcl):
    x, y, z = p
    theta = np.arctan2(y, x)

    R_z = np.array([
        [np.cos(-theta), -np.sin(-theta), 0],
        [np.sin(-theta), np.cos(-theta), 0],
        [0, 0, 1]
    ])

    rotated_pcl = (R_z @ pcl.T).T
    return rotated_pcl

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def create_curved_arrow(ax, start, end, n_points=15, color='purple', alpha=0.6):
    # Calculate the midpoint and displacement vector
    mid = (start + end) / 2
    displacement = end - start

    # Create a perpendicular vector for the curve
    if np.allclose(displacement[0:2], 0):
        # If displacement is vertical, use x-axis as reference
        perp = np.array([1.0, 0.0, 0.0])
    else:
        # Create perpendicular vector in xy-plane
        perp = np.array([-displacement[1], displacement[0], 0])
        perp = perp / np.linalg.norm(perp)

    # Generate points along the curve
    t = np.linspace(0, 1, n_points)
    curve_height = np.linalg.norm(displacement) * 0.3

    points = np.zeros((n_points, 3))
    for i, ti in enumerate(t):
        # Quadratic Bezier curve
        curve_point = (1 - ti) ** 2 * start + \
                      2 * (1 - ti) * ti * (mid + curve_height * perp) + \
                      ti ** 2 * end
        points[i] = curve_point

    # Create arrows along the curve
    arrows = []
    for i in range(len(points) - 1):
        if i % 2 == 0:  # Add arrow every other segment to avoid crowding
            arrow = Arrow3D(
                [points[i][0], points[i + 1][0]],
                [points[i][1], points[i + 1][1]],
                [points[i][2], points[i + 1][2]],
                mutation_scale=30,
                lw=1.5,
                # lw=5,
                arrowstyle='-|>',
                color=color,
                alpha=alpha
            )
            ax.add_artist(arrow)
            arrows.append(arrow)

    return arrows


def plot_point_cloud(pcls, titles, colors, labels):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # Create figure with GridSpec to control spacing
    fig = plt.figure(figsize=(7.2, 7.2))

    # Create a black background for the borders
    fig.patch.set_facecolor('black')

    # Create GridSpec with larger gaps between subplots
    gs = fig.add_gridspec(2, 2,
                          hspace=0.005,  # Spacing between rows
                          wspace=0.005,  # Spacing between columns
                          left=0.0025,  # Left margin
                          right=0.9975,  # Right margin
                          top=0.9975,  # Top margin
                          bottom=0.0025)  # Bottom margin

    # Turn off constrained layout
    plt.rcParams['figure.constrained_layout.use'] = False

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    face_colors = ['red', 'blue', 'green']
    face_alpha = 0.15

    legend_elements = []

    for idx, (pcl, title) in enumerate(zip(pcls, titles)):
        # Create subplot using GridSpec
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col], projection='3d')

        # Set white background for each subplot
        ax.set_facecolor('white')

        # Make sure the axis background is also white
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        # Remove padding within each subplot
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # Reduce spacing around the actual plot
        ax.dist = 10  # Adjust camera distance

        if idx == 0:
            for i in range(2, 5):
                point = pcl[i]
                ax.scatter(*point, c='gray', s=100, edgecolor='k', alpha=0.8)
                ax.scatter(*(pcl[0]), c='black', s=100, edgecolor='k', alpha=0.8)
                if i > 0:
                    ax.plot([pcl[0, 0], point[0]],
                            [pcl[0, 1], point[1]],
                            [pcl[0, 2], point[2]],
                            c='gray', linestyle='--', alpha=0.5, linewidth=3)
        else:
            if idx == 1:
                legend_elements = [plt.scatter([], [], c=color, s=50, edgecolor='k', alpha=0.8, label=label)
                                   for color, label in zip(colors, labels)]
            labels[1] = 'M'
            for i, (point, color, label) in enumerate(zip(pcl, colors, labels)):
                ax.scatter(*point, c=color, s=100, edgecolor='k', alpha=0.8)

                if i > 0:
                    ax.text(point[0], point[1], point[2] + 0.4,
                            f' {label}',
                            fontsize=SMALL_SIZE,
                            color='black',
                            weight='extra bold',
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            path_effects=[withStroke(linewidth=3, foreground='white')])
                    ax.plot([pcl[0, 0], point[0]],
                            [pcl[0, 1], point[1]],
                            [pcl[0, 2], point[2]],
                            c=color, linestyle='--', alpha=0.75, linewidth=3)

            if idx == 2:
                create_curved_arrow(
                    ax,
                    pcls[1][1],
                    pcl[1],
                    color='red',
                    alpha=0.6
                )
            elif idx == 3:
                create_curved_arrow(
                    ax,
                    pcls[2][2],
                    pcl[2],
                    color='green',
                    alpha=0.6
                )

        if len(pcl) >= 3:
            last_three = pcl[-3:, :]
            for h in range(3):
                j = (h + 1) % 3
                ax.plot([last_three[h][0], last_three[j][0]],
                        [last_three[h][1], last_three[j][1]],
                        [last_three[h][2], last_three[j][2]],
                        c='black', linestyle='-', alpha=1, linewidth=1)

        # Just add panel letters without titles
        panel_letters = ['A', 'B', 'C', 'D']
        ax.text2D(0.05, 0.9, panel_letters[idx],
                  color='black',
                  transform=ax.transAxes,
                  fontsize=MEDIUM_SIZE,
                  fontweight='bold',
                  bbox=dict(
                      facecolor='white',
                      edgecolor='none',
                      alpha=1,
                      pad=0
                  ))

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        ax.set_zlim([0, 8])
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=25, azim=45)

    plt.show()

def main():
    points = generate_random_points()
    o = np.zeros(3)
    sizes = np.linalg.norm(points,axis=1)
    size_sorted_indices = np.argsort(sizes)
    size_x_sorted_indices = np.argsort(points[:,0])

    new_size = [sizes[size_sorted_indices[2-i]] for i in range(3)]
    new_size = np.clip(new_size,0,7)
    p1 = points[size_x_sorted_indices[0],:].copy()
    p1[0]=0
    p1 *= (new_size[0] / np.linalg.norm(p1) )
    p2 = points[size_x_sorted_indices[1],:].copy()
    p2 *= (new_size[1] / np.linalg.norm(p2))
    p3 = points[size_x_sorted_indices[2],:].copy()
    p3 *= (new_size[2] / np.linalg.norm(p3))

    M = np.mean([p1,p2,p3], axis=0)
    pcl_1 = np.vstack([o, M, p1, p2, p3])
    pcl_2 = rotate_point_cloud_to_align(M.copy(), pcl_1.copy())
    pcl_3 = rotate_point_z_to_x_axis((pcl_2[2, :]).copy(), pcl_2.copy())

    pcl_list = [pcl_1, pcl_1, pcl_2, pcl_3]
    titles = ["Original Point Cloud", "Map To Canonical Order", "Rotated M onto Z-axis", "Largest Norm onto XZ Plane"]
    colors = ["black", "red", "green", "blue", "orange"]
    labels = ["Point_Of_interest", "Center_Of_Mass (M)", "A", "B", "C"]

    plot_point_cloud(pcl_list, titles, colors, labels)


import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def visualize_point_cloud():
    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Sample 10 points uniformly from [-1,1] x [-1,1]
    num_points = 10
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)

    # Calculate z according to z = x^2 + y^2
    z = x ** 2 + y ** 2

    # Combine into point cloud array
    orig_pcl = np.column_stack((x, y, z))

    # Calculate plot limits with small padding
    padding = 0.05  # Small padding for better visualization
    x_min, x_max = orig_pcl[:, 0].min() - padding, orig_pcl[:, 0].max() + padding
    y_min, y_max = orig_pcl[:, 1].min() - padding, orig_pcl[:, 1].max() + padding
    z_min, z_max = orig_pcl[:, 2].min() - padding, orig_pcl[:, 2].max() + padding

    # 2. Plot original point cloud with points only
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(orig_pcl[:, 0], orig_pcl[:, 1], orig_pcl[:, 2],
                c='crimson', s=100)

    # Set view limits
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_zlim([z_min, z_max])

    # Adjust viewing angle for better visualization
    ax1.view_init(elev=20, azim=45)

    # Remove all axes and grid elements
    ax1.set_axis_off()
    plt.show()

    # 3. Plot point cloud with edges between all points
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plot points
    ax2.scatter(orig_pcl[:, 0], orig_pcl[:, 1], orig_pcl[:, 2],
                c='crimson', s=100)

    # Plot edges between all pairs of points
    for (i, j) in combinations(range(num_points), 2):
        point1 = orig_pcl[i]
        point2 = orig_pcl[j]
        ax2.plot([point1[0], point2[0]],
                 [point1[1], point2[1]],
                 [point1[2], point2[2]],
                 color='steelblue', alpha=0.5, linewidth=1)

    # Set view limits
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_zlim([z_min, z_max])

    # Adjust viewing angle for better visualization
    ax2.view_init(elev=20, azim=45)

    # Remove all axes and grid elements
    ax2.set_axis_off()
    plt.show()


def plot_surfaces():
    def create_grid(x_range=(-2, 2), y_range=(-2, 2), points=100):
        x = np.linspace(x_range[0], x_range[1], points)
        y = np.linspace(y_range[0], y_range[1], points)
        return np.meshgrid(x, y)

    def saddle_ridge(x, y):
        # return np.sin(x) + np.cos(y)
        return x ** 2 - 2*y ** 2

    def ridge(x, y):
        return -np.exp(-0.5 * x ** 2)

    def peak(x, y):
        return -((x ** 2 + y ** 2) / 4)

    def minimal_surface(x, y):
        # return np.sin(x) * np.cosh(y)
        return x ** 2 - y ** 2

    def plane(x, y):
        return np.zeros_like(x)

    def saddle_valley(x, y):
        # return x ** 2 - y ** 2
        return 2* x ** 2 - y ** 2

    def valley(x, y):
        return np.exp(-0.5 * x ** 2)

    def pit(x, y):
        return (x ** 2 + y ** 2) / 4

    # Create figure with gridspec
    fig = plt.figure(figsize=(15, 15))

    # Create GridSpec with different row and column sizes
    gs = gridspec.GridSpec(4, 4, height_ratios=[0.2, 1, 1, 1], width_ratios=[0.2, 1, 1, 1], figure=fig)

    # Add padding between plots
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.075, hspace=0.075)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0, hspace=0.0)

    # Define header texts
    k_headers = ['K < 0', 'K = 0', 'K > 0']
    h_headers = ['H < 0', 'H = 0', 'H > 0']

    # Add K values at the top with borders
    for i, text in enumerate(k_headers, 1):
        ax = fig.add_subplot(gs[0, i])
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=50, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(3)

    # Add H values on the left with borders
    for i, text in enumerate(h_headers, 1):
        ax = fig.add_subplot(gs[i, 0])
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=50, fontweight='bold', rotation=90)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(3)

    # Add corner cell
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.set_xticks([])
    ax_corner.set_yticks([])
    for spine in ax_corner.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(3)

    # Define different colormaps for variety
    first = 'plasma'
    second = 'autumn'
    third = 'winter'
    colormaps = [first, second, third,first, second, third,first, second, third]
    # Define the functions and their titles for each subplot
    surface_functions = [
        (saddle_ridge, 'A', (1, 1)),
        (ridge, 'B', (1, 2)),
        (peak, 'C', (1, 3)),
        (minimal_surface, 'D', (2, 1)),
        (plane, 'E', (2, 2)),
        (None, 'Not possible', (2, 3)),
        (saddle_valley, 'F', (3, 1)),
        (valley, 'G', (3, 2)),
        (pit, 'H', (3, 3))
    ]

    # Create the surface plots
    for idx, (func, title, pos) in enumerate(surface_functions):
        if func is not None:
            ax = fig.add_subplot(gs[pos[0], pos[1]], projection='3d')
            X, Y = create_grid()
            Z = func(X, Y)

            # Create the surface plot with different colormaps
            surf = ax.plot_surface(X, Y, Z, cmap=colormaps[idx], antialiased=True)

            # Customize the plot
            # ax.set_title(title, pad=0, fontsize=50, x=0.075,y=0.925, color='purple', fontweight=1000)
            ax.set_title(title, pad=0, fontsize=50, x=0.9,y=0.075, color='gray', fontweight=1000)
            ax.view_init(elev=30, azim=45)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Add black border
            for spine in ax.spines.values():
                spine.set_visible(False)
                spine.set_color('black')
                spine.set_linewidth(1)
            rect = plt.Rectangle(
                (ax.get_position().x0, ax.get_position().y0),
                ax.get_position().width,
                ax.get_position().height,
                transform=fig.transFigure,
                color='black',
                linewidth=3,
                fill=False
            )
            fig.patches.append(rect)
        else:
            # For the "Not possible" case
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            ax.text(0.5, 0.5, 'Not possible',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=50)
            ax.set_xticks([])
            ax.set_yticks([])
            # Add black border
            for spine in ax.spines.values():
                spine.set_visible(False)
                spine.set_color('black')
                spine.set_linewidth(1)

    # Add a black border around the entire figure
    border_rect = plt.Rectangle(
        (0, 0), 1, 1, transform=fig.transFigure,
        color='black', linewidth=3, fill=False
    )
    fig.patches.append(border_rect)
    plt.show()
if __name__ == "__main__":
    # visualize_point_cloud()
    # a = [main() for i in range(1)]
    # a = [main() for i in range(50)]
    # a = [main() for i in range(60)]
    plot_surfaces()