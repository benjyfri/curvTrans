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

    new_file_path_train = "train_surfaces.h5"
    new_file_path_test = "test_surfaces.h5"
    with h5py.File(new_file_path_train, "w") as new_hdf5_train_file:
        point_clouds_group = new_hdf5_train_file.create_group("point_clouds")
        counter = 0
        for gau_cur in [0, 1, -1]:
            for mean_cur in [0, 1, -1]:
                if gau_cur == 1 and mean_cur == 0:
                    continue
                for k in range(5000):
                    a, b, c, d, e, _, H, K = createFunction(gaussian_curv=gau_cur, mean_curv=mean_cur)
                    point_cloud = samplePoints(a, b, c, d, e, count=20)
                    point_clouds_group.create_dataset(f"point_cloud_{counter}", data=point_cloud)
                    point_clouds_group[f"point_cloud_{counter}"].attrs['a'] = a
                    point_clouds_group[f"point_cloud_{counter}"].attrs['b'] = b
                    point_clouds_group[f"point_cloud_{counter}"].attrs['c'] = c
                    point_clouds_group[f"point_cloud_{counter}"].attrs['d'] = d
                    point_clouds_group[f"point_cloud_{counter}"].attrs['e'] = e
                    point_clouds_group[f"point_cloud_{counter}"].attrs['H'] = H
                    point_clouds_group[f"point_cloud_{counter}"].attrs['K'] = K
                    point_clouds_group[f"point_cloud_{counter}"].attrs['class'] = counter // 5000


                    # patch_file_name = train_path + f'\\{counter}.npy'
                    # info_name = train_path + f'\\{counter}_info.npy'
                    # np.save(patch_file_name, point_cloud)
                    # np.save(info_name, np.array([a , b , c , d , e , H , K , gau_cur , mean_cur]))
                    counter += 1
                    if counter % 500 == 0:
                        print(f'Counter is = {counter}')

    with h5py.File(new_file_path_test, "w") as new_hdf5_test_file:
        point_clouds_group = new_hdf5_test_file.create_group("point_clouds")
        counter = 0
        for gau_cur in [0, 1, -1]:
            for mean_cur in [0, 1, -1]:
                if gau_cur == 1 and mean_cur == 0:
                    continue
                for k in range(500):
                    a, b, c, d, e, _, H, K = createFunction(gaussian_curv=gau_cur, mean_curv=mean_cur)
                    point_cloud = samplePoints(a, b, c, d, e, count=20)
                    point_clouds_group.create_dataset(f"point_cloud_{counter}", data=point_cloud)
                    point_clouds_group[f"point_cloud_{counter}"].attrs['a'] = a
                    point_clouds_group[f"point_cloud_{counter}"].attrs['b'] = b
                    point_clouds_group[f"point_cloud_{counter}"].attrs['c'] = c
                    point_clouds_group[f"point_cloud_{counter}"].attrs['d'] = d
                    point_clouds_group[f"point_cloud_{counter}"].attrs['e'] = e
                    point_clouds_group[f"point_cloud_{counter}"].attrs['H'] = H
                    point_clouds_group[f"point_cloud_{counter}"].attrs['K'] = K
                    point_clouds_group[f"point_cloud_{counter}"].attrs['class'] = counter // 500

                    # patch_file_name = train_path + f'\\{counter}.npy'
                    # info_name = train_path + f'\\{counter}_info.npy'
                    # np.save(patch_file_name, point_cloud)
                    # np.save(info_name, np.array([a , b , c , d , e , H , K , gau_cur , mean_cur]))
                    counter += 1
                    if counter % 500 == 0:
                        print(f'Counter is = {counter}')
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


def plotFunc(a, b, c, d, e):
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

    # Plot the 21 sampled points on the surface
    sampled_points = samplePoints(a, b, c, d, e, count=20)
    for i, point in enumerate(sampled_points):
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                                   mode='markers+text', text=[f'{i + 1}'],
                                   marker=dict(size=25, color='yellow'), name='Point Cloud'),)

    # Show the plot
    fig.show()
if __name__ == '__main__':
    createDataSet()
    # for i in [0,1,-1]:
    #     for j in [0,1,-1]:
    #         if i==1 and j==0:
    #             continue
    #         sum = 0
    #         for k in range(1000):
    #             a, b, c, d, e , count, H , K  = createFunction(gaussian_curv=i, mean_curv=j)
    #             sum += count
    #         print(f'Avg runs for gaussian_curv={i}, mean_curv={j} = {sum/1000} ')
    #         plotFunc(a,b,c,d,e)

    print("yay")