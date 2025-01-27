from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke

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


def create_curved_arrow(ax, start, end, n_points=20, color='purple', alpha=0.6):
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
                mutation_scale=10,
                lw=1.5,
                arrowstyle='-|>',
                color=color,
                alpha=alpha
            )
            ax.add_artist(arrow)
            arrows.append(arrow)

    return arrows



def plot_point_cloud_old(pcls, titles, colors, labels):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    fig = plt.figure(figsize=(7.2, 7.2))

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    face_colors = ['red', 'blue', 'green']
    face_alpha = 0.15

    # Create legend handles
    legend_elements = []

    for idx, (pcl, title) in enumerate(zip(pcls, titles)):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        if idx == 0:
            for i in range(2, 5):
                point = pcl[i]
                ax.scatter(*point, c='gray', s=50, edgecolor='k', alpha=0.8)
                ax.scatter(*(pcl[0]), c='black', s=50, edgecolor='k', alpha=0.8)
                if i > 0:
                    ax.plot([pcl[0, 0], point[0]],
                            [pcl[0, 1], point[1]],
                            [pcl[0, 2], point[2]],
                            c='gray', linestyle='--', alpha=0.5, linewidth=2)

        else:
            # Create scatter points for legend
            if idx == 1:  # Only create legend elements once
                legend_elements = [plt.scatter([], [], c=color, s=50, edgecolor='k', alpha=0.8, label=label)
                                   for color, label in zip(colors, labels)]
            labels[1] = 'M'
            for i, (point, color, label) in enumerate(zip(pcl, colors, labels)):
                ax.scatter(*point, c=color, s=50, edgecolor='k', alpha=0.8)

                if i > 0:
                    ax.text(point[0], point[1], point[2] + 0.4,
                            f' {label}',
                            fontsize=10,
                            color='black',
                            weight='extra bold',
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            path_effects=[withStroke(linewidth=3, foreground='white')])
                    ax.plot([pcl[0, 0], point[0]],
                            [pcl[0, 1], point[1]],
                            [pcl[0, 2], point[2]],
                            c=color, linestyle='--', alpha=0.5, linewidth=2)

            # Add rotation arrows for specific plots
            if idx == 2:  # Third plot - M rotation to z-axis
                create_curved_arrow(
                    ax,
                    pcls[1][1],  # Original M position
                    pcl[1],  # New M position
                    color='red',
                    alpha=0.6
                )

            elif idx == 3:  # Fourth plot - A rotation around z-axis
                create_curved_arrow(
                    ax,
                    pcls[2][2],  # Position of A before z-axis rotation
                    pcl[2],  # Final position of A
                    color='green',
                    alpha=0.6
                )

        if len(pcl) >= 3:  # Make sure there are at least 3 points
            last_three = pcl[-3:,:]  # Get the last three points
            # Draw lines between all pairs of the last three points
            for h in range(3):
                j = (h + 1) % 3  # This cycles through 0,1,2 to connect all points
                ax.plot([last_three[h][0], last_three[j][0]],
                        [last_three[h][1], last_three[j][1]],
                        [last_three[h][2], last_three[j][2]],
                        c='black', linestyle='-', alpha=1, linewidth=1)

        panel_letters = ['A', 'B', 'C', 'D']
        ax.set_title(f'{panel_letters[idx]} | {title}',
                     pad=15,
                     fontweight='bold',
                     bbox=dict(
                         facecolor='white',
                         edgecolor='gray',
                         alpha=0.8,
                         pad=5
                     ),
                     fontsize=MEDIUM_SIZE,
                     color='darkblue',
                     loc='left'  # This aligns the title to the left
                     )
        # ax.set_xlabel('X', labelpad=5)
        # ax.set_ylabel('Y', labelpad=5)
        # ax.set_zlabel('Z', labelpad=5)

        ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_xlim([0, 5])
        ax.set_ylim([0, 4])
        ax.set_zlim([0, 8])
        ax.set_box_aspect([1, 1, 1])

        ax.view_init(elev=25, azim=45)


    if idx > 0:
        fig.legend(
            handles=legend_elements,
            loc='center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(labels),
            fontsize=8
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, wspace=0.0, hspace=0.3)  # Changed wspace from 0.3 to 0.15
    plt.savefig('point_cloud_visualization.png',
                format='png',
                bbox_inches='tight',
                dpi=300)
    plt.show()


def plot_point_cloud(pcls, titles, colors, labels):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # Create figure with minimal margins
    fig = plt.figure(figsize=(7.2, 7.2))

    # Remove default margins
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.05
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.0
    plt.rcParams['figure.constrained_layout.hspace'] = 0.05
    plt.rcParams['figure.constrained_layout.wspace'] = 0.0

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

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
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

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
                ax.scatter(*point, c='gray', s=50, edgecolor='k', alpha=0.8)
                ax.scatter(*(pcl[0]), c='black', s=50, edgecolor='k', alpha=0.8)
                if i > 0:
                    ax.plot([pcl[0, 0], point[0]],
                            [pcl[0, 1], point[1]],
                            [pcl[0, 2], point[2]],
                            c='gray', linestyle='--', alpha=0.5, linewidth=2)
        else:
            if idx == 1:
                legend_elements = [plt.scatter([], [], c=color, s=50, edgecolor='k', alpha=0.8, label=label)
                                   for color, label in zip(colors, labels)]
            labels[1] = 'M'
            for i, (point, color, label) in enumerate(zip(pcl, colors, labels)):
                ax.scatter(*point, c=color, s=50, edgecolor='k', alpha=0.8)

                if i > 0:
                    ax.text(point[0], point[1], point[2] + 0.4,
                            f' {label}',
                            fontsize=10,
                            color='black',
                            weight='extra bold',
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            path_effects=[withStroke(linewidth=3, foreground='white')])
                    ax.plot([pcl[0, 0], point[0]],
                            [pcl[0, 1], point[1]],
                            [pcl[0, 2], point[2]],
                            c=color, linestyle='--', alpha=0.5, linewidth=2)

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

        panel_letters = ['A', 'B', 'C', 'D']
        ax.set_title(f'{panel_letters[idx]} | {title}',
                     pad=2,  # Reduced padding
                     fontweight='bold',
                     bbox=dict(
                         facecolor='white',
                         edgecolor='gray',
                         alpha=0.8,
                         pad=2  # Reduced padding
                     ),
                     fontsize=MEDIUM_SIZE,
                     color='darkblue',
                     loc='center'
                     )

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 4])
        ax.set_zlim([0, 8])
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=25, azim=45)

    if idx > 0:
        fig.legend(
            handles=legend_elements,
            loc='center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(labels),
            fontsize=8
        )

    # Use tight_layout with minimal padding
    plt.tight_layout(pad=0.1, h_pad=0.2, w_pad=0.2)

    # Save with minimal borders
    plt.savefig('point_cloud_visualization.png',
                format='png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300)
    plt.show()
def main():
    points = generate_random_points()
    M = np.mean(points, axis=0)
    o = np.zeros(3)
    size_sorted = np.argsort(np.linalg.norm(points,axis=1))
    p1 = points[size_sorted[2],:].copy()
    p2 = points[size_sorted[1],:].copy()
    p3 = points[size_sorted[0],:].copy()

    pcl_1 = np.vstack([o, M, p1, p2, p3])
    pcl_2 = rotate_point_cloud_to_align(M.copy(), pcl_1.copy())
    pcl_3 = rotate_point_z_to_x_axis((pcl_2[2, :]).copy(), pcl_2.copy())

    pcl_list = [pcl_1, pcl_1, pcl_2, pcl_3]
    titles = ["Original Point Cloud", "Map To Canonical Order", "Rotated M onto Z-axis", "Largest Norm onto XZ Plane"]
    colors = ["black", "red", "green", "blue", "orange"]
    labels = ["Point_Of_interest", "Center_Of_Mass (M)", "A", "B", "C"]

    plot_point_cloud(pcl_list, titles, colors, labels)


if __name__ == "__main__":
    # a = [main() for i in range(1)]
    a = [main() for i in range(40)]