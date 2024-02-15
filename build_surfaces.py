import os
import random
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import h5py

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
    At the moment I'm only creating "mountains" as in Lidar that seems to be mor likely
    to scan a protruding mountain than a valley or potato chip
    :return:
    '''
    #create folders
    train_path = os.path.join(os.getcwd(), 'train_surfaces')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.join(os.getcwd(), 'test_surfaces')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    new_file_path_train = "train_surfaces_40.h5"
    new_file_path_test = "test_surfaces_40.h5"
    with h5py.File(new_file_path_train, "w") as new_hdf5_train_file:
        point_clouds_group = new_hdf5_train_file.create_group("point_clouds")
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=0, label=0, counter=0, amount_of_pcl=10000,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=1, mean_curv=1, label=1, counter=10000, amount_of_pcl=5000,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, counter=15000, amount_of_pcl=5000,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=1, label=2, counter=20000, amount_of_pcl=5000,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=2, counter=25000, amount_of_pcl=5000,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=-1, mean_curv=-33, label=3, counter=30000, amount_of_pcl=10000,
                     size_of_pcl=40)

    with h5py.File(new_file_path_test, "w") as new_hdf5_test_file:
        point_clouds_group = new_hdf5_test_file.create_group("point_clouds")
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=0, label=0, counter=0, amount_of_pcl=1000,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=1, mean_curv=1, label=1, counter=1000, amount_of_pcl=500,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, counter=1500, amount_of_pcl=500,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=1, label=2, counter=2000, amount_of_pcl=500,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=2, counter=2500, amount_of_pcl=500,
                     size_of_pcl=40)
        addDataToSet(point_clouds_group, gaussian_curv=-1, mean_curv=-33, label=3, counter=3000, amount_of_pcl=1000,
                     size_of_pcl=40)

def addDataToSet(point_clouds_group, gaussian_curv, mean_curv, label, counter, amount_of_pcl, size_of_pcl=40):
    for k in range(amount_of_pcl):
        a, b, c, d, e, _, H, K = createFunction(gaussian_curv=gaussian_curv, mean_curv=mean_curv)
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
    # plotFunc(a, b, c, d, e, point_cloud)
def createFunction(gaussian_curv, mean_curv, epsilon=0.05):
    if gaussian_curv==1 and mean_curv==0:
        raise ValueError("gaussian_curv==1 and mean_curv==0 is impossible")
    okFunc = False
    count = 0
    while not okFunc:
        okFunc=True
        count += 1
        a, b, c, d, e = [random.uniform(-5, 5) for _ in range(5)]
        K = (4*(a*b)-((c**2))) / (1 + d**2 + e**2)
        H = (2*a*(1 + c**2)-2*d*e*c +2*b*(1 + d**2)) / ( ( (d**2) + (e**2) + 1 )**1.5)
        # zero gaussian curve
        if gaussian_curv==0:
            if abs(K) > epsilon:
                okFunc=False
                continue
        # positive gaussian curv
        if gaussian_curv==1:
            if K < (10*epsilon):
                okFunc=False
                continue
        # negative gaussian curv
        if gaussian_curv==-1:
            if K > -(10*epsilon):
                okFunc=False
                continue

        # zero mean curve
        if mean_curv==0:
            if abs(H) > epsilon:
                okFunc=False
                continue
        # positive mean curv
        if mean_curv==1:
            if H < (10*epsilon):
                okFunc=False
                continue
        # negative mean curv
        if mean_curv==-1:
            if H > -(10*epsilon):
                okFunc=False
                continue

    return a, b, c, d, e, count , H , K

import plotly.graph_objects as go

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

    # Customize layout
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      title="Sampled Points")

    # Show the plot
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
def plotMultiplePcls(parameter_sets, index=1):
    # Create a Plotly figure
    fig = go.Figure()
    import matplotlib.cm as cm

    # Number of colors needed
    num_colors = len(parameter_sets)  # You can adjust this based on the number of point clouds

    # Create an array of colors using the 'viridis' colormap
    # colors = cm.Paired(np.linspace(0, 1, num_colors))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
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
if __name__ == '__main__':
    createDataSet()
    # parameter_sets = [
    #     (-0.317246927541051, -1.4981434283801311, -0.8403160811352457, 0.45210163040038864, 4.989508427603763),
    #
    #     (3.6184079799238305, 0.7546575719059714, 3.3933402363050025, 1.4572342266597076, -4.891390159508835),
    #
    #     (-0.25023974028386053, -4.153200676255113, 2.0461446225436664, -4.655450773639295, 0.7272193519327672),
    #
    #     (3.838197172376347, 4.353654426721661, 2.9045692276212574, -3.943710916344255, 1.1583435215106501),
    #
    #     (-4.056479577616709, -2.2233117868694774, -3.0415162535894478, -3.2630328492056746, -2.682355605186225),
    #
    #     (1.2879827469937988, 2.192264473460156, -4.331727514839182, 0.06295771570294662, -0.010133464803817382)
    # ]
    # for i in range(1,6):
    #     plotMultiplePcls(parameter_sets, i)
    print("yay")