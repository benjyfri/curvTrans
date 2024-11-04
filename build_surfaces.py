import os
import random
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import h5py
import matplotlib.pyplot as plt

def samplePoints(a, b, c, d, e, count):
    def surface_function(x, y):
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y

    # Generate random points within the range [-1, 1] for both x and y
    x_samples = np.random.uniform(-1, 1, count)
    y_samples = np.random.uniform(-1, 1, count)

    # Evaluate the surface function at the random points
    z_samples = surface_function(x_samples, y_samples)

    # Create an array with the sampled points
    sampled_points = np.column_stack((x_samples, y_samples, z_samples))

    # Concatenate the centroid [0, 0, 0] to the beginning of the array
    centroid = np.array([[0, 0, 0]])
    sampled_points_with_centroid = np.concatenate((centroid, sampled_points), axis=0)

    return sampled_points_with_centroid
def createDataSet():
    '''
    A
    :return:
    '''
    #create folders
    train_path = os.path.join(os.getcwd(), 'train_surfaces')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.join(os.getcwd(), 'test_surfaces')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    new_file_path_train = "train_surfaces_with_corners_very_mild.h5"
    new_file_path_test = "test_surfaces_with_corners_very_mild.h5"
    with h5py.File(new_file_path_train, "w") as new_hdf5_train_file:
        point_clouds_group = new_hdf5_train_file.create_group("point_clouds")
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=0, label=0, counter=0, amount_of_pcl=10000,
                     size_of_pcl=40)
        print(f'Finished train flat surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=1, mean_curv=1, label=1, counter=10000, amount_of_pcl=5000,
                     size_of_pcl=40)
        print(f'Finished train parabolic peak surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, counter=15000, amount_of_pcl=5000,
                     size_of_pcl=40)
        print(f'Finished train parabolic pit surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=1, label=2, counter=20000, amount_of_pcl=5000,
                     size_of_pcl=40)
        print(f'Finished train ridge surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=2, counter=25000, amount_of_pcl=5000,
                     size_of_pcl=40)
        print(f'Finished train valley surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=-1, mean_curv=-33, label=3, counter=30000, amount_of_pcl=10000,
                     size_of_pcl=40)
        print(f'Finished train saddle surfaces')
        addDataCornersToSet(point_clouds_group, angle=30, label=4, counter=40000, amount_of_pcl=10000, size_of_pcl=40)
        print(f'Finished train 30 angle surfaces')
        addDataCornersToSet(point_clouds_group, angle=90, label=5, counter=50000, amount_of_pcl=10000, size_of_pcl=40)
        print(f'Finished train 90 angle surfaces')
        addDataCornersToSet(point_clouds_group, angle=150, label=6, counter=60000, amount_of_pcl=10000, size_of_pcl=40)
        print(f'Finished train 150 angle surfaces')
        addDataCornersToSet(point_clouds_group, angle=360, label=7, counter=70000, amount_of_pcl=10000, size_of_pcl=40)
        print(f'Finished train room corner surfaces')

    with h5py.File(new_file_path_test, "w") as new_hdf5_test_file:
        point_clouds_group = new_hdf5_test_file.create_group("point_clouds")
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=0, label=0, counter=0, amount_of_pcl=1000,
                     size_of_pcl=40)
        print(f'Finished test flat surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=1, mean_curv=1, label=1, counter=1000, amount_of_pcl=500,
                     size_of_pcl=40)
        print(f'Finished test parabolic peak surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, counter=1500, amount_of_pcl=500,
                     size_of_pcl=40)
        print(f'Finished test parabolic pit surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=1, label=2, counter=2000, amount_of_pcl=500,
                     size_of_pcl=40)
        print(f'Finished test ridge surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=2, counter=2500, amount_of_pcl=500,
                     size_of_pcl=40)
        print(f'Finished test valley surfaces')
        addDataToSet(point_clouds_group, gaussian_curv=-1, mean_curv=-33, label=3, counter=3000, amount_of_pcl=1000,
                     size_of_pcl=40)
        print(f'Finished test saddle surfaces')
        addDataCornersToSet(point_clouds_group, angle=30, label=4, counter=4000, amount_of_pcl=1000, size_of_pcl=40)
        print(f'Finished test 30 angle surfaces')
        addDataCornersToSet(point_clouds_group, angle=90, label=5, counter=5000, amount_of_pcl=1000, size_of_pcl=40)
        print(f'Finished test 90 angle surfaces')
        addDataCornersToSet(point_clouds_group, angle=150, label=6, counter=6000, amount_of_pcl=1000, size_of_pcl=40)
        print(f'Finished test 150 angle surfaces')
        addDataCornersToSet(point_clouds_group, angle=360, label=7, counter=7000, amount_of_pcl=1000, size_of_pcl=40)
        print(f'Finished test room corner surfaces')

def addDataToSet(point_clouds_group, gaussian_curv, mean_curv, label, counter, amount_of_pcl, size_of_pcl=40):
    for k in range(amount_of_pcl):
        a, b, c, d, e, _, H, K = createFunction(gaussian_curv=gaussian_curv, mean_curv=mean_curv, boundary=0.7, epsilon=0.4)
        point_cloud = samplePoints(a, b, c, d, e, count=size_of_pcl)
        point_clouds_group.create_dataset(f"point_cloud_{counter+k}", data=point_cloud)
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['a'] = a
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['b'] = b
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['c'] = c
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['d'] = d
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['e'] = e
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['H'] = H
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['K'] = K
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['class'] = label
def addDataCornersToSet(point_clouds_group,angle, label, counter, amount_of_pcl, size_of_pcl=40):
    sampling_cur = generate_surfaces_angles_and_sample
    # room corner situation
    if angle ==360:
        sampling_cur=generate_room_corner_with_points
    for k in range(amount_of_pcl):
        rand_angle = np.random.uniform(angle-10, angle+10)
        point_cloud = sampling_cur(size_of_pcl, rand_angle)
        # point_clouds_group.create_dataset(f"point_cloud_{counter+k}", data=point_cloud)
        point_clouds_group.create_dataset(f"point_cloud_{counter+k}", data=np.array([0,0,0]).reshape(1,3))
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['a'] = rand_angle
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['b'] = rand_angle
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['c'] = rand_angle
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['d'] = rand_angle
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['e'] = rand_angle
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['H'] = rand_angle
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['K'] = rand_angle
        point_clouds_group[f"point_cloud_{counter+k}"].attrs['class'] = label

def createFunction(gaussian_curv, mean_curv, boundary=3, epsilon=0.05):
    if gaussian_curv==1 and mean_curv==0:
        raise ValueError("gaussian_curv==1 and mean_curv==0 is impossible")
    okFunc = False
    count = 0
    while not okFunc:
        okFunc=True
        count += 1
        a, b, c, d, e = np.random.uniform(-1, 1, 5)
        K = (4*(a*b)-((c**2))) / ((1 + d**2 + e**2)**2)
        H = (a*(1 + e**2)-d*e*c +b*(1 + d**2)) / ( ( (d**2) + (e**2) + 1 )**1.5)


        # Not to steep
        if abs(H)> 3 or abs(K)>3:
            okFunc = False
            continue
        # zero gaussian curve
        if gaussian_curv==0:
            if abs(K) > epsilon:
                okFunc=False
                continue
        # positive gaussian curv
        if gaussian_curv==1:
            if K < boundary:
                okFunc=False
                continue
        # negative gaussian curv
        if gaussian_curv==-1:
            if K > -(boundary):
                okFunc=False
                continue

        # zero mean curve
        if mean_curv==0:
            if abs(H) > epsilon:
                okFunc=False
                continue
        # positive mean curv
        if mean_curv==1:
            if H < (boundary):
                okFunc=False
                continue
        # negative mean curv
        if mean_curv==-1:
            if H > -(boundary):
                okFunc=False
                continue

    return a, b, c, d, e, count , H , K


def plotPcl(a, b, c, d, e, sample_count=40):
    # Generate 40 random points within the specified range
    sampled_points = samplePoints(a, b, c, d, e, count=sample_count)

    # Create 3D scatter plot for sampled points using Plotly Express
    fig = go.Figure()

    # Add sampled points to the figure
    fig.add_trace(go.Scatter3d(x=[sampled_points[0][0]],
                               y=[sampled_points[0][1]],
                               z=[sampled_points[0][2]],
                               mode='markers',
                               marker=dict(size=8, color='blue'),
                               text=['1'],
                               textposition='middle center',
                               name='First Point'))

    fig.add_trace(go.Scatter3d(x=[point[0] for point in sampled_points[1:]],
                               y=[point[1] for point in sampled_points[1:]],
                               z=[point[2] for point in sampled_points[1:]],
                               mode='markers',
                               marker=dict(size=8, color='red'),
                               text=[f'{i + 2}' for i in range(sample_count - 1)],
                               textposition='middle center',
                               name='Rest of Points'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(r=20, l=10, b=10, t=50)
    )
    fig.show()


def plotFunc(a, b, c, d, e,sampled_points):
    # Create a grid of points for the surface
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, y)

    # Compute the surface using the generated coefficients
    z = a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y

    # Compute the distance from each point to the origin
    distance_to_origin = np.sqrt(x**2 + y**2)

    # Create a mask for points within a radius of 0.25 from the origin
    mask = distance_to_origin <= 0.25

    # Create 3D surface plot using Plotly Express
    fig = px.scatter_3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), color=mask.flatten(),
                        color_continuous_scale=['blue', 'red'], title="Generated Surface",
                        labels={'x': 'X', 'y': 'Y', 'z': 'Z'}, range_color=[0, 1])

    for i, point in enumerate(sampled_points):
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                                   mode='markers+text', text=[f'{i + 1}'],
                                   marker=dict(size=25, color='yellow'), name='Point Cloud'),)

    # Show the plot
    fig.show()
def plotMultiplePcls(parameter_sets,names=[], index=1):
    # Create a Plotly figure
    fig = go.Figure()
    import matplotlib.cm as cm

    # Number of colors needed
    num_colors = len(parameter_sets)  # You can adjust this based on the number of point clouds

    # Create an array of colors using the 'viridis' colormap
    # colors = cm.Paired(np.linspace(0, 1, num_colors))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
    if len(names)==0:
        names = ['surface', 'valley', 'ridge', 'bowl', 'mountain', 'saddle']
    # Iterate over each set of parameters
    for i, params in enumerate(parameter_sets):
        if i != index and i>0:
            continue
        # Generate sampled points for the current set of parameters
        sampled_points = samplePoints(*params, count=40)

        color = colors[i]
        name = names[i]
        fig.add_trace(go.Scatter3d(x=[point[0] for point in sampled_points],
                                   y=[point[1] for point in sampled_points],
                                   z=[point[2] for point in sampled_points],
                                   mode='markers',
                                   marker=dict(size=8, color=color),
                                   text=[f'{j + 1}' for j in range(40)],
                                   textposition='middle center',
                                   name=f'{name}'))

    fig.add_trace(go.Scatter3d(x=[sampled_points[0][0]],
                               y=[sampled_points[0][1]],
                               z=[sampled_points[0][2]],
                               mode='markers',
                               marker=dict(size=8, color='yellow'),
                               text=['1'],
                               textposition='middle center',
                               name='First Point'))
    # Customize layout
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      title="Sampled Point Clouds in Different Colors",
                      showlegend=True)

    # Show the plot
    fig.show()


def fit_surface_quadratic_constrained(points):
    """
    Fits a quadratic surface constrained to f = 0 to a centered point cloud.

    Args:
      points: numpy array of shape (N, 3) representing the point cloud.

    Returns:
      numpy array of shape (5,) representing the surface coefficients
        [a, b, c, d, e], where:
          z = a * x**2 + b * y**2 + c * x * y + d * x + e * y
    """

    # Center the points around the mean
    centroid = points[0,:]
    centered_points = points - centroid

    # Design matrix without f term
    X = np.c_[centered_points[:, 0] ** 2, centered_points[:, 1] ** 2,
              centered_points[:, 0] * centered_points[:, 1],
    centered_points[:, 0], centered_points[:, 1]]

    # Extract z-coordinates as target vector
    z = centered_points[:, 2]

    # Solve the linear system with f coefficient constrained to 0
    coeffs = np.linalg.lstsq(X, z, rcond=None)[0]

    a, b, c, d, e = coeffs

    K = (4 * (a * b) - ((c ** 2))) / ((1 + d ** 2 + e ** 2) ** 2)
    H = (a * (1 + e ** 2) - d * e * c + b * (1 + d ** 2)) / (((d ** 2) + (e ** 2) + 1) ** 1.5)
    return a, b, c, d, e, K, H
def accuracyHKdependingOnNumOfPoints(sigma=0):
    H_acc_mean =[]
    H_acc_median =[]
    K_acc_mean =[]
    K_acc_median =[]
    func_loss_mean = []
    func_loss_median = []
    a_values = []
    b_values = []
    c_values = []
    d_values = []
    e_values = []
    options = [(0, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, -33)]
    for i in range(5, 55, 5):
        print(f'i : {i}')
        cur_loss_H = []
        cur_loss_K = []
        cur_func_loss = []
        a_temp = []
        b_temp = []
        c_temp = []
        d_temp = []
        e_temp = []
        for j in range(50):
            if j % 10 == 0:
                print(f'j: {j}')
            random_setup = random.choice(options)
            a1, b1, c1, d1, e1, _, H, K = createFunction(gaussian_curv=random_setup[0], mean_curv=random_setup[1],
                                                         boundary=5, epsilon=0.05)
            point_cloud = samplePoints(a1, b1, c1, d1, e1, count=i)
            noised_point_cloud = point_cloud + np.random.normal(loc=0, scale=sigma, size=point_cloud.shape)
            noised_point_cloud_centered = noised_point_cloud - noised_point_cloud[0,:]
            a2, b2, c2, d2, e2, K2, H2 = fit_surface_quadratic_constrained(noised_point_cloud_centered)

            def surface_function(x, y):
                return a2 * x ** 2 + b2 * y ** 2 + c2 * x * y + d2 * x + e2 * y

            # Calculate the loss for each point in the original point cloud
            losses = [abs(surface_function(x, y) - h) for (x, y, h) in point_cloud[1:,:]]
            cur_func_loss.append(np.mean(losses))


            cur_loss_H.append(np.linalg.norm(H - H2))
            cur_loss_K.append(np.linalg.norm(K - K2))
            a_temp.append(np.linalg.norm(a1-a2))
            b_temp.append(np.linalg.norm(b1-b2))
            c_temp.append(np.linalg.norm(c1-c2))
            d_temp.append(np.linalg.norm(d1-d2))
            e_temp.append(np.linalg.norm(e1-e2))
        H_acc_mean.append(np.mean(cur_loss_H))
        H_acc_median.append(np.median(cur_loss_H))
        K_acc_mean.append(np.mean(cur_loss_K))
        K_acc_median.append(np.median(cur_loss_K))
        func_loss_mean.append(np.mean(cur_func_loss))
        func_loss_median.append(np.median(cur_func_loss))
        a_values.append(np.mean(a_temp))
        b_values.append(np.mean(b_temp))
        c_values.append(np.mean(c_temp))
        d_values.append(np.mean(d_temp))
        e_values.append(np.mean(e_temp))

    plt.figure(figsize=(15, 18))  # Increased height to accommodate more subplots

    # Adjust subplot parameters to increase vertical spacing
    plt.subplots_adjust(hspace=0.5)  # Increased vertical spacing between subplots

    plt.subplot(4, 1, 1)
    plt.plot(range(5, 55, 5), H_acc_mean, label='Mean Loss H')
    plt.plot(range(5, 55, 5), H_acc_median, label='Median Loss H')
    plt.title(f'Accuracy of H Depending on Number of Points; std = {sigma}')
    plt.xlabel('Number of Points')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(range(5, 55, 5), K_acc_mean, label='Mean Loss K')
    plt.plot(range(5, 55, 5), K_acc_median, label='Median Loss K')
    plt.title(f'Accuracy of K Depending on Number of Points; std = {sigma}')
    plt.xlabel('Number of Points')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(range(5, 55, 5), a_values, label='Mean Loss a')
    plt.plot(range(5, 55, 5), b_values, label='Mean Loss b')
    plt.plot(range(5, 55, 5), c_values, label='Mean Loss c')
    plt.plot(range(5, 55, 5), d_values, label='Mean Loss d')
    plt.plot(range(5, 55, 5), e_values, label='Mean Loss e')
    plt.title(f'Mean Coefficients a, b, c, d, e Depending on Number of Points; std = {sigma}')
    plt.xlabel('Number of Points')
    plt.ylabel('Mean Coefficient Value')
    # plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(range(5, 55, 5), func_loss_mean, label='Mean Loss func')
    plt.plot(range(5, 55, 5), func_loss_median, label='Median Loss func')
    plt.title(f'Func_loss Depending on Number of Points; std = {sigma}')
    plt.xlabel('Number of Points')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()  # Tight layout adjustment
    plt.show()

def testNoiseEffect(sigma=0):
    options = [(0, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, -33)]
    names = ["plane", "pit", "peak", "valley", "ridge", "saddle"]
    for setup, name in zip(options, names):
        H_list = []
        K_list = []
        H2_list = []
        K2_list = []
        a1, b1, c1, d1, e1, _, H, K = createFunction(gaussian_curv=setup[0], mean_curv=setup[1],
                                                     boundary=5, epsilon=0.05)
        point_cloud = samplePoints(a1, b1, c1, d1, e1, count=40)
        for _ in range(30):
            noised_point_cloud = point_cloud + np.random.normal(loc=0, scale=sigma, size=point_cloud.shape)
            noised_point_cloud_centered = noised_point_cloud - noised_point_cloud[0, :]
            a2, b2, c2, d2, e2, K2, H2 = fit_surface_quadratic_constrained(noised_point_cloud_centered)
            H_list.append(H)
            K_list.append(K)
            H2_list.append(H2)
            K2_list.append(K2)
        fig = plt.figure(figsize=(16, 12))

        # Plot H and H2
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(H_list, label='H')
        ax1.plot(H2_list, label='H2')
        ax1.set_xlabel('Setup Index')
        ax1.set_ylabel('Values')
        ax1.set_title(f'Comparison of H and H2_estimate')
        ax1.legend()

        # Plot K and K2
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(K_list, label='K')
        ax2.plot(K2_list, label='K2')
        ax2.set_xlabel('Setup Index')
        ax2.set_ylabel('Values')
        ax2.set_title(f'Comparison of K and K2_estimate')
        ax2.legend()

        # Plot 3D function
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = a1 * X ** 2 + b1 * Y ** 2 + c1 * X * Y + d1 * X + e1 * Y
        ax3.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Original Function Plot')

        # Plot clean point cloud
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='.', s=150)
        ax4.scatter(noised_point_cloud_centered[:, 0], noised_point_cloud_centered[:, 1], noised_point_cloud_centered[:, 2], c='r', marker='.', s=150)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_title('Original (blue) and noised (red) pcl')

        plt.tight_layout()
        plt.show()


def random_rotation(point_cloud):
    # Generate random rotation angles around x, y, and z axes
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    # Rotation matrices around x, y, and z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    # Combine rotation matrices
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Apply rotation to point cloud
    rotated_point_cloud = np.dot(point_cloud, R.T)
    # plot_point_clouds(point_cloud, rotated_point_cloud)
    is_rotation = np.allclose(np.eye(3), np.dot(R, R.T))
    if not is_rotation:
        raise ValueError("not a rotation")
    return rotated_point_cloud
def plot_point_clouds(point_cloud1, point_cloud2):
    """
    Plot two point clouds in an interactive 3D plot with Plotly.

    Args:
        point_cloud1 (np.ndarray): First point cloud of shape (41, 3)
        point_cloud2 (np.ndarray): Second point cloud of shape (41, 3)
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2],
        mode='markers', marker=dict(color='red'), name='Point Cloud 1'
    ))

    fig.add_trace(go.Scatter3d(
        x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
        mode='markers', marker=dict(color='blue'), name='Point Cloud 2'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig.show()

def generate_room_corner_with_points(N, angle=None):
    N1, N2, N3 = np.random.multinomial(N-3, [1/3, 1/3, 1/3]) + np.array([1,1,1])

    x_coords1 = np.random.uniform(0, 1, N)
    y_coords1 = np.random.uniform(0, 1, N)

    center = np.array([0, 0, 0])
    points1 = np.stack((np.random.uniform(0, 1, N1), np.random.uniform(0, 1, N1), np.zeros(N1)), axis=-1)
    points2 = np.stack((np.zeros(N2), np.random.uniform(0, 1, N2), -np.random.uniform(0, 1, N2)), axis=-1)
    points3 = np.stack((np.random.uniform(0, 1, N3), np.zeros(N3), -np.random.uniform(0, 1, N3)), axis=-1)
    points = np.vstack((center, points1,points2,points3))
    return points
def generate_surfaces_angles_and_sample(N, angle):
    # 1. Generate a random angle between 0 and 30 degrees
    angle_rad = np.radians((180 - angle)/2)

    # 2. Compute the slopes (m1 and m2) for the surfaces
    m1 = np.tan(angle_rad)  # slope for the left surface (x < 0)
    m2 = -m1  # slope for the right surface (x >= 0)

    # 3. Generate N random points in the square [-1, 1] x [-1, 1]
    x_coords = np.random.uniform(-1, 1, N)
    y_coords = np.random.uniform(-1, 1, N)

    # 4. Calculate the corresponding z values based on the surfaces
    z_coords = np.where(x_coords < 0, m1 * x_coords, m2 * x_coords)
    # z_coords = np.abs(x_coords)

    # 5. Stack the points into a single array
    points = np.stack((x_coords, y_coords, z_coords), axis=-1)
    center = np.array([0,0,0])
    points = np.vstack((center,points))
    return points

if __name__ == '__main__':
    createDataSet()
    print("yay")