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
    x_samples = np.random.uniform(-2, 2, count)
    y_samples = np.random.uniform(-2, 2, count)

    # Evaluate the surface function at the random points
    z_samples = surface_function(x_samples, y_samples)

    # Create an array with the sampled points
    sampled_points = np.column_stack((x_samples, y_samples, z_samples))

    # Concatenate the centroid [0, 0, 0] to the beginning of the array
    centroid = np.array([[0, 0, 0]])
    sampled_points_with_centroid = np.concatenate((centroid, sampled_points), axis=0)

    return sampled_points_with_centroid

def sampleHalfSpacePoints(a, b, c, d, e, count):
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
    center_point_idx = np.argsort(np.linalg.norm(sampled_points_with_centroid, axis=1))[-1]
    sampled_points_with_centroid = sampled_points_with_centroid - sampled_points_with_centroid[center_point_idx, :]
    return sampled_points_with_centroid

def updateDataSet(label_to_update=4,new_file_path_train = "train_surfaces_1X1.h5", new_file_path_test = "test_surfaces_1X1.h5"):
    """
    Creates or updates datasets in an HDF5 file based on a specific label to update.
    """

    def addIfMatchingLabel(group, gaussian_curv, mean_curv, label, **kwargs):
        """
        Adds or updates datasets only if the label matches the specified label_to_update.
        """
        if label == label_to_update:
            addDataToSet(group, gaussian_curv, mean_curv, label, update=True, **kwargs)

    with h5py.File(new_file_path_train, "r+") as new_hdf5_train_file:
        point_clouds_group = new_hdf5_train_file.get("point_clouds")
        if point_clouds_group is None:
            point_clouds_group = new_hdf5_train_file.create_group("point_clouds")

        # Train data generation for each label
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=0, label=0, boundary=3, epsilon=0.5, counter=0, amount_of_pcl=10000)
        print(f'Finished train flat surfaces')

        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=1, label=1, boundary=3, epsilon=1, counter=10000, amount_of_pcl=2500)
        print(f'Finished train parabolic peak surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=12500, amount_of_pcl=2500)
        print(f'Finished train parabolic pit surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=15000, amount_of_pcl=2500, angle=1)
        print(f'Finished train parabolic CORNERS')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=17500, amount_of_pcl=2500, radius=1)
        print(f'Finished train parabolic SPHERES')

        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=2, boundary=3, epsilon=1, counter=20000, amount_of_pcl=2500)
        print(f'Finished train ridge surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=2, boundary=3, epsilon=1, counter=22500, amount_of_pcl=2500)
        print(f'Finished train valley surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=2, boundary=3, epsilon=1, counter=25000, amount_of_pcl=2500, angle=1)
        print(f'Finished train ridge ANGLES')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=2, boundary=3, epsilon=1, counter=27500, amount_of_pcl=2500, radius=1)
        print(f'Finished train valley Cylinders')

        addIfMatchingLabel(point_clouds_group, gaussian_curv=-1, mean_curv=-33, label=3, boundary=4.5, epsilon=1, counter=30000, amount_of_pcl=10000)
        print(f'Finished train saddle surfaces')

        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=0, label=4, boundary=3, epsilon=1, counter=40000, amount_of_pcl=2500)
        print(f'Finished train HALFSPACE flat surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=42500, amount_of_pcl=625)
        print(f'Finished train HALFSPACE parabolic peak surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=43125, amount_of_pcl=625)
        print(f'Finished train HALFSPACE parabolic pit surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=43750, amount_of_pcl=625, angle=1, edge=1)
        print(f'Finished train HALFSPACE parabolic CORNERS')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=44375, amount_of_pcl=625, radius=1, edge=1)
        print(f'Finished train HALFSPACE parabolic SPHERES')
        # addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=43750, amount_of_pcl=625)
        # print(f'Finished train HALFSPACE parabolic CORNERS')
        # addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=44375, amount_of_pcl=625)
        # print(f'Finished train HALFSPACE parabolic SPHERES')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=45000, amount_of_pcl=625)
        print(f'Finished train HALFSPACE ridge surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=45625, amount_of_pcl=625)
        print(f'Finished train HALFSPACE valley surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=46250, amount_of_pcl=625, angle=1, edge=2)
        print(f'Finished train HALFSPACE ridge ANGLES')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=46875, amount_of_pcl=625, radius=1, edge=2)
        print(f'Finished train HALFSPACE valley CYLINDERS')
        # addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=46250, amount_of_pcl=625)
        # print(f'Finished train HALFSPACE ridge ANGLES')
        # addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=46875, amount_of_pcl=625)
        # print(f'Finished train HALFSPACE valley CYLINDERS')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=-1, mean_curv=-33, label=4, boundary=3, epsilon=1, counter=47500, amount_of_pcl=2500)
        print(f'Finished train HALFSPACE saddle surfaces')

    with h5py.File(new_file_path_test, "r+") as new_hdf5_test_file:
        point_clouds_group = new_hdf5_test_file.get("point_clouds")
        if point_clouds_group is None:
            point_clouds_group = new_hdf5_test_file.create_group("point_clouds")

        # Test data generation for each label
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=0, label=0, boundary=3, epsilon=0.5, counter=0, amount_of_pcl=1000)
        print(f'Finished test flat surfaces')

        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=1, label=1, boundary=3, epsilon=1, counter=1000, amount_of_pcl=250)
        print(f'Finished test parabolic peak surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=1250, amount_of_pcl=250)
        print(f'Finished test parabolic pit surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=1500, amount_of_pcl=250, angle=1)
        print(f'Finished test parabolic CORNERS')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=1750, amount_of_pcl=250, radius=1)
        print(f'Finished test parabolic SPHERES')

        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=2, boundary=3, epsilon=1, counter=2000, amount_of_pcl=250)
        print(f'Finished test ridge surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=2, boundary=3, epsilon=1, counter=2250, amount_of_pcl=250)
        print(f'Finished test valley surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=2, boundary=3, epsilon=1, counter=2500, amount_of_pcl=250, angle=1)
        print(f'Finished test ridge ANGLES')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=2, boundary=3, epsilon=1, counter=2750, amount_of_pcl=250, radius=1)
        print(f'Finished test valley Cylinders')

        addIfMatchingLabel(point_clouds_group, gaussian_curv=-1, mean_curv=-33, label=3, boundary=4.5, epsilon=1, counter=3000, amount_of_pcl=1000)
        print(f'Finished test saddle surfaces')

        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=0, label=4, boundary=3, epsilon=1, counter=4000, amount_of_pcl=250)
        print(f'Finished test HALFSPACE flat surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4250, amount_of_pcl=63)
        print(f'Finished test HALFSPACE parabolic peak surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4313, amount_of_pcl=63)
        print(f'Finished test HALFSPACE parabolic pit surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4376, amount_of_pcl=63, angle=1, edge=1)
        print(f'Finished test HALFSPACE parabolic CORNERS')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4439, amount_of_pcl=63, radius=1, edge=1)
        print(f'Finished test HALFSPACE parabolic SPHERES')
        # addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4376, amount_of_pcl=63)
        # print(f'Finished test HALFSPACE parabolic CORNERS')
        # addIfMatchingLabel(point_clouds_group, gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4439, amount_of_pcl=63)
        # print(f'Finished test HALFSPACE parabolic SPHERES')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4502, amount_of_pcl=63)
        print(f'Finished test HALFSPACE ridge surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4565, amount_of_pcl=63)
        print(f'Finished test HALFSPACE valley surfaces')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4628, amount_of_pcl=63, angle=1, edge=2)
        print(f'Finished test HALFSPACE ridge ANGLES')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4691, amount_of_pcl=63, radius=1, edge=2)
        print(f'Finished test HALFSPACE valley CYLINDERS')
        # addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4628, amount_of_pcl=63)
        # print(f'Finished test HALFSPACE ridge ANGLES')
        # addIfMatchingLabel(point_clouds_group, gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4691, amount_of_pcl=63)
        # print(f'Finished test HALFSPACE valley CYLINDERS')
        addIfMatchingLabel(point_clouds_group, gaussian_curv=-1, mean_curv=-33, label=4, boundary=3, epsilon=1, counter=4754, amount_of_pcl=250)
        print(f'Finished test HALFSPACE saddle surfaces')

def createDataSetOld():
    new_file_path_train = "train_surfaces_1X1.h5"
    new_file_path_test = "test_surfaces_1X1.h5"
    #for each shpae
    boundaries =[0.5, 0.5,0,5, 1.8]
    epsilons = [0.05, 0.2, 0.2, 0.2]
    with h5py.File(new_file_path_train, "w") as new_hdf5_train_file:
        point_clouds_group = new_hdf5_train_file.create_group("point_clouds")

        # Train data generation for each label
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=0, label=0, boundary=3, epsilon=0.5, counter=0, amount_of_pcl=10000)
        print(f'Finished train flat surfaces')

        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=1, label=1, boundary=3, epsilon=1, counter=10000,
                     amount_of_pcl=2500)
        print(f'Finished train parabolic peak surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=12500,
                     amount_of_pcl=2500)
        print(f'Finished train parabolic pit surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=15000,
                     amount_of_pcl=2500, angle=1)
        print(f'Finished train parabolic CORNERS')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=17500,
                     amount_of_pcl=2500, radius=1)
        print(f'Finished train parabolic SPHERES')

        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=1, label=2, boundary=3, epsilon=1, counter=20000,
                     amount_of_pcl=2500)
        print(f'Finished train ridge surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=-1, label=2, boundary=3, epsilon=1, counter=22500,
                     amount_of_pcl=2500)
        print(f'Finished train valley surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=1, label=2, boundary=3, epsilon=1, counter=25000,
                     amount_of_pcl=2500, angle=1)
        print(f'Finished train ridge ANGLES')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=-1, label=2, boundary=3, epsilon=1, counter=27500,
                     amount_of_pcl=2500, radius=1)
        print(f'Finished train valley Cylinders')

        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=-1, mean_curv=-33, label=3, boundary=4.5, epsilon=1, counter=30000,
                     amount_of_pcl=10000)
        print(f'Finished train saddle surfaces')

        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=0, label=4, boundary=3, epsilon=1, counter=40000,
                     amount_of_pcl=2500)
        print(f'Finished train HALFSPACE flat surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=42500, amount_of_pcl=625)
        print(f'Finished train HALFSPACE parabolic peak surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=43125,
                     amount_of_pcl=625)
        print(f'Finished train HALFSPACE parabolic pit surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=43750, amount_of_pcl=625,
                     angle=1, edge=1)
        print(f'Finished train HALFSPACE parabolic CORNERS')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=44375,
                     amount_of_pcl=625, radius=1, edge=1)
        print(f'Finished train HALFSPACE parabolic SPHERES') 
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=45000, amount_of_pcl=625)
        print(f'Finished train HALFSPACE ridge surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=45625,
                     amount_of_pcl=625)
        print(f'Finished train HALFSPACE valley surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=46250, amount_of_pcl=625,
                     angle=1, edge=2)
        print(f'Finished train HALFSPACE ridge ANGLES')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=46875,
                     amount_of_pcl=625, radius=1, edge=2)
        print(f'Finished train HALFSPACE valley CYLINDERS')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=-1, mean_curv=-33, label=4, boundary=3, epsilon=1, counter=47500,
                     amount_of_pcl=2500)
        print(f'Finished train HALFSPACE saddle surfaces')

    with h5py.File(new_file_path_test, "w") as new_hdf5_test_file:
        point_clouds_group = new_hdf5_test_file.create_group("point_clouds")
        # Test data generation for each label
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=0, label=0, boundary=3, epsilon=0.5, counter=0, amount_of_pcl=1000)
        print(f'Finished test flat surfaces')

        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=1, label=1, boundary=3, epsilon=1, counter=1000, amount_of_pcl=250)
        print(f'Finished test parabolic peak surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=1250, amount_of_pcl=250)
        print(f'Finished test parabolic pit surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=1500, amount_of_pcl=250,
                     angle=1)
        print(f'Finished test parabolic CORNERS')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=1, boundary=3, epsilon=1, counter=1750, amount_of_pcl=250,
                     radius=1)
        print(f'Finished test parabolic SPHERES')

        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=1, label=2, boundary=3, epsilon=1, counter=2000, amount_of_pcl=250)
        print(f'Finished test ridge surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=-1, label=2, boundary=3, epsilon=1, counter=2250, amount_of_pcl=250)
        print(f'Finished test valley surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=1, label=2, boundary=3, epsilon=1, counter=2500, amount_of_pcl=250,
                     angle=1)
        print(f'Finished test ridge ANGLES')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=-1, label=2, boundary=3, epsilon=1, counter=2750, amount_of_pcl=250,
                     radius=1)
        print(f'Finished test valley Cylinders')

        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=-1, mean_curv=-33, label=3, boundary=4.5, epsilon=1, counter=3000,
                     amount_of_pcl=1000)
        print(f'Finished test saddle surfaces')

        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=0, label=4, boundary=3, epsilon=1, counter=4000, amount_of_pcl=250)
        print(f'Finished test HALFSPACE flat surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4250, amount_of_pcl=63)
        print(f'Finished test HALFSPACE parabolic peak surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4313, amount_of_pcl=63)
        print(f'Finished test HALFSPACE parabolic pit surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4376, amount_of_pcl=63,
                     angle=1, edge=1)
        print(f'Finished test HALFSPACE parabolic CORNERS')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=1, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4439, amount_of_pcl=63,
                     radius=1, edge=1)
        print(f'Finished test HALFSPACE parabolic SPHERES')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4502, amount_of_pcl=63)
        print(f'Finished test HALFSPACE ridge surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4565, amount_of_pcl=63)
        print(f'Finished test HALFSPACE valley surfaces')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=1, label=4, boundary=3, epsilon=1, counter=4628, amount_of_pcl=63,
                     angle=1, edge=2)
        print(f'Finished test HALFSPACE ridge ANGLES')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=0, mean_curv=-1, label=4, boundary=3, epsilon=1, counter=4691, amount_of_pcl=63,
                     radius=1, edge=2)
        print(f'Finished test HALFSPACE valley CYLINDERS')
        addDataToSet(point_clouds_group=point_clouds_group,gaussian_curv=-1, mean_curv=-33, label=4, boundary=3, epsilon=1, counter=4754,
                     amount_of_pcl=250)
        print(f'Finished test HALFSPACE saddle surfaces')


def addDataToSet(point_clouds_group, gaussian_curv, mean_curv, label, counter, amount_of_pcl, boundary, epsilon, angle=0, radius=0, edge=0, update=False):
    max_curv = 5
    default_max_angle = 120
    default_min_angle = 30
    constant = max_curv / (2 * np.cos(np.radians( default_min_angle ) / 2)) + 0.05 # make sure that highest angle is 30 + add epsilon for corner cases
    for k in range(amount_of_pcl):
        if angle!=0 or radius!=0:
            a=b=c=d=e=H=K = 0
            if angle>0:
                if label==1 or edge==1:
                    # make sure the height of the pyramid is not too low and not too high (using the connection between R and h and the angles of the tip of the pyramid)
                    # pythagoras thm on height of pyr and height of face:  h^2 + 0.25R^2 = 0.75R^2 * 1 / (tan(alpha/2))^2; tan(rise_aang) = h / R
                    # because parabola has curvature 2*a we enforce same values for rise of pyramid at 0.5
                    boundary_rise_angle_rad = np.arctan( 0.25 * boundary)
                    max_curv_rise_angle_rad = np.arctan( 0.25 * max_curv)
                    min_rise_angle_default = (np.pi - np.radians(default_max_angle)) / 2
                    max_rise_angle_default = (np.pi - np.radians(default_min_angle)) / 2
                    min_rise_angle = max(boundary_rise_angle_rad, min_rise_angle_default)
                    max_rise_angle = min(max_curv_rise_angle_rad, max_rise_angle_default)
                    max_pyr_angle = 2 * (np.arctan(1 / ((0.25 + np.tan(min_rise_angle)**2 ) / 0.75 )))
                    min_pyr_angle = 2 * (np.arctan(1 / ((0.25 + np.tan(max_rise_angle)**2 ) / 0.75 )))

                    if min_pyr_angle < 0:
                        print("fix tis pyramid!!")
                        exit(-1)

                    angle_rad = np.random.uniform(min_pyr_angle, max_pyr_angle)
                    cur_curve = np.sqrt(2*np.pi - 3*angle_rad)
                    k1 = k2 = cur_curve
                    angle = np.degrees(angle_rad)
                if label==2 or edge==2:
                    max_angle_rad = np.clip(2 * np.arccos(boundary / (2 * constant)), np.radians(default_min_angle), np.radians(default_max_angle))
                    min_angle_rad = np.clip(2 * np.arccos(max_curv / (2 * constant)), np.radians(default_min_angle), np.radians(default_max_angle))
                    angle_rad = np.random.uniform(min_angle_rad, max_angle_rad)
                    cur_curve = constant * (2 * np.cos(angle_rad / 2))
                    k1 = cur_curve
                    k2 = 0
                    angle = np.degrees(angle_rad)
            if radius > 0:
                cur_curve = np.random.uniform(boundary ,max_curv)
                radius = 1 / cur_curve
                if label==1 or edge==1:
                    k1 = k2 = cur_curve
                if label==2 or edge==2:
                    k1 = cur_curve
                    k2 = 0
        else:
            a, b, c, d, e, _, k1, k2 = createFunction(gaussian_curv=gaussian_curv, mean_curv=mean_curv, boundary=boundary, epsilon=epsilon, max_curv=max_curv)
        dataset_name = f"point_cloud_{counter + k}"
        if update == True:
            del point_clouds_group[dataset_name]
        point_cloud = point_clouds_group.create_dataset(dataset_name, data=np.array([0, 0, 0]).reshape(1, 3))
        point_cloud.attrs['a'] = a
        point_cloud.attrs['b'] = b
        point_cloud.attrs['c'] = c
        point_cloud.attrs['d'] = d
        point_cloud.attrs['e'] = e
        point_cloud.attrs['k1'] = k1
        point_cloud.attrs['k2'] = k2
        point_cloud.attrs['angle'] = angle
        point_cloud.attrs['radius'] = radius
        point_cloud.attrs['class'] = label
        point_cloud.attrs['edge'] = edge

def createFunction(gaussian_curv, mean_curv, max_curv, boundary=3, epsilon=0.25):
    if gaussian_curv==1 and mean_curv==0:
        raise ValueError("gaussian_curv==1 and mean_curv==0 is impossible")
    okFunc = False
    count = 0
    while not okFunc:
        okFunc=True
        count += 1
        # a, b, c, d, e = np.random.uniform(-1.3, 1.3, 5)
        a, b, c, d, e = np.random.uniform(-3, 3, 5)
        K = (4*(a*b)-((c**2))) / ((1 + d**2 + e**2)**2)
        H = (a*(1 + e**2)-d*e*c +b*(1 + d**2)) / ( ( (d**2) + (e**2) + 1 )**1.5)

        discriminant = H ** 2 - K
        k1 = H + np.sqrt(discriminant)
        k2 = H - np.sqrt(discriminant)


        temp_max = k1 if abs(k1) > abs(k2) else k2
        temp_min = k1 if abs(k1) < abs(k2) else k2

        # Not to steep
        if abs(temp_max) > max_curv:
            okFunc = False
            continue

        # zero gaussian curve --> either plane or ridge/valley
        if gaussian_curv==0:
            if (abs(temp_min) > epsilon):
                okFunc=False
                continue
            # positive mean curv
            if mean_curv == 0:
                if (abs(temp_max) > epsilon):
                    okFunc = False
                    continue
            # positive mean curv
            if mean_curv == 1:
                if temp_max < (boundary):
                    okFunc = False
                    continue
            # negative mean curv
            if mean_curv == -1:
                if temp_max > -(boundary):
                    okFunc = False
                    continue

        # non-zero gaussian curve --> either parabola or saddle
        else:
            if (abs(temp_min) < boundary):
                okFunc = False
                continue
            # positive gaussian curv
            if gaussian_curv==1:
                if K < 0:
                    okFunc=False
                    continue
            # negative gaussian curv
            if gaussian_curv==-1:
                if K > 0:
                    okFunc=False
                    continue

    return a, b, c, d, e, count , k1, k2


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


def plotFunc(a, b, c, d, e,sampled_points=None):
    # Create a grid of points for the surface
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
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

    x_range = np.ptp(x)  # Peak-to-peak range for X
    y_range = np.ptp(y)  # Peak-to-peak range for Y
    z_range = np.ptp(z)
    max_range = max(x_range, y_range, z_range)
    aspect_ratio = dict(
        x=x_range / max_range,
        y=y_range / max_range,
        z=z_range / max_range
    )
    # Determine aspect ratio based on ranges
    max_range = max(x_range, y_range, z_range)
    aspect_ratio = dict(
        x=x_range / max_range,
        y=y_range / max_range,
        z=z_range / max_range
    )

    if sampled_points is not None:
        for i, point in enumerate(sampled_points):
            fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                                       mode='markers+text', text=[f'{i + 1}'],
                                       marker=dict(size=25, color='yellow'), name='Point Cloud'),)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        aspectmode='manual',
        aspectratio=aspect_ratio
        ),
        margin=dict(r=20, l=10, b=10, t=50)
    )
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

def rotatePCLToCanonical(point_cloud):
    # Calculate the covariance matrix
    cov_matrix = np.cov(point_cloud, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Find the smallest eigenvector corresponding to the smallest eigenvalue
    normal_at_centroid = eigenvectors[:, np.argmin(eigenvalues)]
    normal_at_centroid /= np.linalg.norm(normal_at_centroid)

    # Calculate the rotation matrix to align the normal with the z-axis
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal_at_centroid, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal_at_centroid, z_axis)

    if s == 0:  # Already aligned with z-axis
        rotation_matrix = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))

    # Rotate the point cloud and centroid
    rotated_point_cloud = point_cloud @ rotation_matrix.T

    return rotated_point_cloud
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
    discriminant = H ** 2 - K
    k1 = H + np.sqrt(discriminant)
    k2 = H - np.sqrt(discriminant)
    return a, b, c, d, e, K, H, k1, k2

def calc_curvatures(a=0,b=0,c=0,d=0,e=0):
    K = (4 * (a * b) - ((c ** 2))) / ((1 + d ** 2 + e ** 2) ** 2)
    H = (a * (1 + e ** 2) - d * e * c + b * (1 + d ** 2)) / (((d ** 2) + (e ** 2) + 1) ** 1.5)
    discriminant = H ** 2 - K
    k1 = H + np.sqrt(discriminant)
    k2 = H - np.sqrt(discriminant)
    return k1, k2, K, H
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
                                                         boundary=5, epsilon=0.25)
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

def plot_point_clouds(point_cloud1, point_cloud2=None):
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
    if point_cloud2 is not None:
        fig.add_trace(go.Scatter3d(
            x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2],
            mode='markers', marker=dict(color='blue'), name='Point Cloud 2'
        ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(color='yellow'), name='Center'
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


if __name__ == '__main__':
    updateDataSet(label_to_update=0)
    # createDataSetOld()
    # updateDataSet(label_to_update=1)
    # updateDataSet(label_to_update=2)
    # updateDataSet(label_to_update=3)
    # updateDataSet(label_to_update=4)
    # createDataSetOld()
    a = 0