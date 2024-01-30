import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __():
    import os
    from os.path import join, isfile, isdir
    import cv2

    import xml.etree.ElementTree as ET
    import matplotlib.pyplot as plt
    return ET, cv2, isdir, isfile, join, os, plt


@app.cell
def __():
    DATA_DIR = '../../Datasets/CUTE80_Dataset'
    return DATA_DIR,


@app.cell
def __(DATA_DIR, join):
    IMAGE_FOLDER = join(DATA_DIR, 'CUTE80')
    ground_truth_file = join(DATA_DIR, 'Groundtruth', 'GroundTruth.xml')
    return IMAGE_FOLDER, ground_truth_file


@app.cell
def __(IMAGE_FOLDER, os):
    im_names = next(os.walk(IMAGE_FOLDER))[2]
    print(im_names[0])
    return im_names,


@app.cell
def __(ET, ground_truth_file):
    # Load the XML file
    tree = ET.parse(ground_truth_file)

    # Get the root element
    root = tree.getroot()

    data = dict()
    class Polygon:
        def __init__(self,xs,ys):
            self.points = []
            xs = xs.split(' ')
            ys = ys.split(' ')
            for (x, y) in zip(xs, ys):
                self.points.append([int(x), int(y)])

    # Iterate over the <Image> elements
    for image in root.findall('Image'):
        # Get the <ImageName> element
        image_name = image.find('ImageName').text
        # print('Image Name:', image_name)
        data[image_name] = list()    
        # Iterate over the <PolygonPoints> elements
        for polygon in image.findall('PolygonPoints'):
            x_points = polygon.get('x')
            y_points = polygon.get('y')
            text = polygon.text
            # print('Polygon Points (x):', x_points)
            # print('Polygon Points (y):', y_points)
            polygon_data = Polygon(x_points, y_points) 
            data[image_name].append(polygon_data)
        # print('---')
    return (
        Polygon,
        data,
        image,
        image_name,
        polygon,
        polygon_data,
        root,
        text,
        tree,
        x_points,
        y_points,
    )


@app.cell
def __():
    image_to_vis = 'image001.jpg'
    return image_to_vis,


@app.cell
def __(IMAGE_FOLDER, cv2, image_to_vis, join):
    cv2_image = cv2.imread(join(IMAGE_FOLDER, image_to_vis))
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) 
    return cv2_image,


@app.cell
def __(cv2_image, plt):
    plt.imshow(cv2_image)
    return


@app.cell
def __(cv2, cv2_image, data, image_to_vis):
    label = data[image_to_vis]
    import copy
    vis_image = copy.deepcopy(cv2_image)
    for each_polygon in label:
        points = each_polygon.points
        points.append(points[0])
        num_point = len(points) - 1
        for i in range(num_point):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            vis_image = cv2.line(vis_image, points[i], points[i+1], (0, 255, 0), 2)

    return (
        copy,
        each_polygon,
        i,
        label,
        num_point,
        points,
        vis_image,
        x1,
        x2,
        y1,
        y2,
    )


@app.cell
def __(plt, vis_image):
    plt.imshow(vis_image)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
