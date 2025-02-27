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


def create_curved_arrow(ax, start, end, n_points=12, color='purple', alpha=0.6):
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
                lw=3,
                arrowstyle='-|>',
                color=color,
                alpha=alpha
            )
            ax.add_artist(arrow)
            arrows.append(arrow)

    return arrows


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke


# Base plotting setup function
def setup_3d_plot():
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 16, 20, 24
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection='3d')

    # Set white background
    ax.set_facecolor('white')
    for pane in [ax.xaxis, ax.yaxis, ax.zaxis]:
        pane.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Remove labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    return fig, ax


def plot_common_elements(ax, pcl, panel_letter):
    # Plot connections between last three points
    if len(pcl) >= 3:
        last_three = pcl[-3:, :]
        for h in range(3):
            j = (h + 1) % 3
            ax.plot([last_three[h][0], last_three[j][0]],
                    [last_three[h][1], last_three[j][1]],
                    [last_three[h][2], last_three[j][2]],
                    c='black', linestyle='-', alpha=1, linewidth=1)

    # Add panel letter
    ax.text2D(0.075, 0.9, panel_letter,
              color='gray',
              transform=ax.transAxes,
              fontsize=20,
              fontweight=1000,
              bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=0))

    # Set common view properties
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 4.5])
    ax.set_ylim([0, 4.5])
    ax.set_zlim([0, 8])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=45)


def plot_original_point_cloud(pcl, panel_letter='A'):
    fig, ax = setup_3d_plot()

    # Plot origin point
    ax.scatter(*(pcl[0]), c='black', s=100, edgecolor='k', alpha=0.8)

    # Plot other points and their connections to origin
    colors = ['gray', 'blue', 'green', 'red']
    for i in range(2, 5):
        point = pcl[i]
        ax.scatter(*point, c=colors[i-1], s=100, edgecolor='k', alpha=0.8)
        ax.plot([pcl[0, 0], point[0]],
                [pcl[0, 1], point[1]],
                [pcl[0, 2], point[2]],
                c='gray', linestyle='--', alpha=0.5, linewidth=3)

    plot_common_elements(ax, pcl, panel_letter)
    plt.tight_layout()
    return fig


def plot_canonical_order(pcl, colors, labels, panel_letter='B'):
    fig, ax = setup_3d_plot()

    # Create legend elements
    legend_elements = [plt.scatter([], [], c=color, s=50, edgecolor='k',
                                   alpha=0.8, label=label)
                       for color, label in zip(colors, labels)]

    # Update center of mass label
    labels[1] = 'M'

    # Plot points and connections
    for i, (point, color, label) in enumerate(zip(pcl, colors, labels)):
        ax.scatter(*point, c=color, s=100, edgecolor='k', alpha=0.8)

        if i > 0:
            # Add point labels
            ax.text(point[0], point[1], point[2] + 0.4,
                    f' {label}',
                    fontsize=16,
                    color='black',
                    weight='extra bold',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    path_effects=[withStroke(linewidth=3, foreground='white')])

            # Add connections to origin
            ax.plot([pcl[0, 0], point[0]],
                    [pcl[0, 1], point[1]],
                    [pcl[0, 2], point[2]],
                    c=color, linestyle='--', alpha=0.75, linewidth=3)

    plot_common_elements(ax, pcl, panel_letter)
    plt.tight_layout()
    return fig


def plot_rotated_m(pcl, colors, labels, prev_pcl, panel_letter='C'):
    fig, ax = setup_3d_plot()

    # Plot points and connections similar to canonical order
    for i, (point, color, label) in enumerate(zip(pcl, colors, labels)):
        ax.scatter(*point, c=color, s=100, edgecolor='k', alpha=0.8)

        if i > 0:
            ax.text(point[0], point[1], point[2] + 0.4,
                    f' {label}',
                    fontsize=16,
                    color='black',
                    weight='extra bold',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    path_effects=[withStroke(linewidth=3, foreground='white')])
            ax.plot([pcl[0, 0], point[0]],
                    [pcl[0, 1], point[1]],
                    [pcl[0, 2], point[2]],
                    c=color, linestyle='--', alpha=0.75, linewidth=3)

    # Add curved arrow
    create_curved_arrow(ax, prev_pcl[1], pcl[1], color='red', alpha=0.4, n_points=12)

    plot_common_elements(ax, pcl, panel_letter)
    plt.tight_layout()
    return fig


def plot_largest_norm(pcl, colors, labels, prev_pcl, panel_letter='D'):
    fig, ax = setup_3d_plot()

    # Plot points and connections
    for i, (point, color, label) in enumerate(zip(pcl, colors, labels)):
        ax.scatter(*point, c=color, s=100, edgecolor='k', alpha=0.8)

        if i > 0:
            ax.text(point[0], point[1], point[2] + 0.4,
                    f' {label}',
                    fontsize=16,
                    color='black',
                    weight='extra bold',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    path_effects=[withStroke(linewidth=3, foreground='white')])
            ax.plot([pcl[0, 0], point[0]],
                    [pcl[0, 1], point[1]],
                    [pcl[0, 2], point[2]],
                    c=color, linestyle='--', alpha=0.75, linewidth=3)

    # Add curved arrow
    create_curved_arrow(ax, prev_pcl[2], pcl[2], color='green', alpha=0.4, n_points=15)

    plot_common_elements(ax, pcl, panel_letter)
    plt.tight_layout()
    return fig


def plot_points_only(pcl, panel_letter='E'):
    fig, ax = setup_3d_plot()

    # Plot only origin and three points (skip center of mass)
    points_to_plot = [pcl[0]] + list(pcl[2:])  # Origin and three points

    # Plot points without any connections
    colors = ['gray', 'blue', 'green', 'red']
    for i, point in enumerate(points_to_plot):
        ax.scatter(*point, c=colors[i], s=100, edgecolor='k', alpha=0.8)

    # Keep the grid and 3D setup, just remove connections
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 4.5])
    ax.set_ylim([0, 4.5])
    ax.set_zlim([0, 8])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=45)

    # Add panel letter
    ax.text2D(0.075, 0.9, panel_letter,
              color='gray',
              transform=ax.transAxes,
              fontsize=20,
              fontweight=1000,
              bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=0))

    plt.tight_layout()
    return fig


def plot_1d_mapping(pcl, values, panel_letter='F'):
    # Create figure with specific size to match other plots
    fig = plt.figure(figsize=(7.2, 7.2))

    # Create subplot layout with appropriate spacing
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 0.5], hspace=0.1)

    # 3D subplot for original points
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.set_facecolor('white')
    for pane in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
        pane.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Plot only the original points in 3D (excluding center of mass)
    points_to_plot = [pcl[0]] + list(pcl[2:])  # Origin and three points
    colors = ['gray', 'blue', 'green', 'red']
    colors = ['gray', 'blue', 'green', 'red']

    for point, color in zip(points_to_plot, colors):
        ax1.scatter(*point, c=color, s=300, edgecolor='k', alpha=0.8)

    # Set 3D plot properties
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim([0, 4.5])
    ax1.set_ylim([0, 4.5])
    ax1.set_zlim([0, 8])
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=25, azim=45)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])

    # 1D subplot
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('white')

    # Plot 1D points with matching colors
    y_positions = np.zeros_like(values)
    values =  values[1:]
    colors =  colors[1:]
    for i, (value, color) in enumerate(zip(values, colors)):
        ax2.scatter(value, y_positions[i], c=color, s=300, edgecolor='k', alpha=0.8)

        # Add value labels
        ax2.annotate(f'{value:.1f}',
                     (value, 0),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     va='bottom')

    # Add horizontal line
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Customize 1D plot
    ax2.set_ylim([-1, 1])
    ax2.set_yticks([])
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Add panel letter
    ax1.text2D(0.075, 0.9, panel_letter,
               color='gray',
               transform=ax1.transAxes,
               fontsize=20,
               fontweight=1000,
               bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=0))

    plt.tight_layout()
    return fig

def plot_point_cloud(pcls, titles, colors, labels):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    figures = []

    # Call specific plotting function for each visualization

    figures.append(plot_points_only(pcls[0]))
    figures.append(plot_original_point_cloud(pcls[0]))
    figures.append(plot_canonical_order(pcls[1], colors, labels))
    figures.append(plot_rotated_m(pcls[2], colors, labels, pcls[1]))
    figures.append(plot_largest_norm(pcls[3], colors, labels, pcls[2]))

    # Create array of points without center of mass for 1D mapping
    mapping_values = [2.1, -3.3, -4.5, 5.0]
    figures.append(plot_1d_mapping(pcls[0], mapping_values))

    # Save figures
    for i, fig in enumerate(figures):
        fig.savefig(f"{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    return figures
def main(i=0):
    # Keep the main function as is, just call the new plot_point_cloud function
    points = np.array([[2.15886298, 5.50488619, 1.5282085],
                       [1.40084327, 1.43710485, 5.28841026],
                       [5.57066292, 3.88908508, 3.49825685]])
    o = np.zeros(3)
    sizes = np.linalg.norm(points, axis=1)
    size_sorted_indices = np.argsort(sizes)
    size_x_sorted_indices = np.argsort(points[:, 0])

    new_size = [sizes[size_sorted_indices[2 - i]] for i in range(3)]
    new_size = np.clip(new_size, 0, 7)
    p1 = points[size_x_sorted_indices[0], :].copy()
    p1[0] = 0
    p1 *= (new_size[0] / np.linalg.norm(p1))
    p2 = points[size_x_sorted_indices[1], :].copy()
    p2 *= (new_size[1] / np.linalg.norm(p2))
    p3 = points[size_x_sorted_indices[2], :].copy()
    p3 *= (new_size[2] / np.linalg.norm(p3))

    M = np.mean([p1, p2, p3], axis=0)
    pcl_1 = np.vstack([o, M, p1, p2, p3])
    pcl_2 = rotate_point_cloud_to_align(M.copy(), pcl_1.copy())
    pcl_3 = rotate_point_z_to_x_axis((pcl_2[2, :]).copy(), pcl_2.copy())

    pcl_list = [pcl_1, pcl_1, pcl_2, pcl_3]
    titles = ["Original Point Cloud", "Map To Canonical Order", "Rotated M onto Z-axis",
              "Largest Norm onto XZ Plane"]
    colors = ["black", "red", "green", "blue", "orange"]
    labels = ["Point_Of_interest", "Center_Of_Mass (M)", "A", "B", "C"]

    figures = plot_point_cloud(pcl_list, titles, colors, labels)


if __name__ == "__main__":
    main()