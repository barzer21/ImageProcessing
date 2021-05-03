# This is a sample Python script.
import cv2 as cv
import numpy as np
import sys
from mpl_toolkits.mplot3d import axes3d
from win32api import GetSystemMetrics
from pylab import *
import imutils


# this function sort sampled frames and return sorted x_coordinate, y_coordinate and frame arrays
def creatCordinate(frame_list):
    index_list = {}
    index_list['x'] = []
    index_list['y'] = []
    index_list['z'] = []

    frame_list_sorted = sorted(frame_list)
    for frame in frame_list_sorted:
        index_list['x'].append(frame_list[frame][0])
        index_list['y'].append(frame_list[frame][1])
        index_list['z'].append(frame)

    return index_list

# this function interpolate the x,y coordinate of the image in a specific frame z according to the coordinates of two
# sampled frames (p1 & p2) before and after frame z
def find_xy(p1, p2, z):

    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if z2 < z1:
        return find_xy(p2, p1, z)

    x = np.interp(z, (z1, z2), (x1, x2))
    y = np.interp(z, (z1, z2), (y1, y2))

    return int(x), int(y)


def main(argv):

    frame_list = {} # dictionary for image coordinates per frame number
    scale_list = {} # dictionary for scale per frame number
    rotate_list = {} # dictionary for rotation angle per frame number
    img = cv.imread(cv.samples.findFile(argv[1]))
    img_h, img_w, img_c = img.shape
    cap = cv.VideoCapture(argv[0])
    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) # Y
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) # X
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) # Z

    # this function sample and show image coordinates on desired frame
    def mouse_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            new_img = cv.resize(img, (int(np.ceil(float(scale) * img_w)), int(np.ceil(float(scale) * img_h))),
                                interpolation=cv.INTER_CUBIC) # scaling the image
            new_img = imutils.rotate_bound(new_img, int(angle)) # rotating the image
            if y + new_img.shape[0] < h and x + new_img.shape[1] < w: # check if new image is within frame boundaries
                # saving the parameters in the dictionaries and showing the new frame
                frame_list[int(input_frame)] = (x, y)
                scale_list[int(input_frame)] = float(scale)
                rotate_list[int(input_frame)] = int(angle)
                frame[y:y + new_img.shape[0], x:x + new_img.shape[1]] = new_img
                cv.imshow('Display window', frame)
                cv.waitKey(5)
            else:
                print("Unvalid position, press again")

    out = True
    while out:
        input_frame = input("Pick frame number between 0 - " + str(n-1) + " : \n")
        if input_frame == "end": # if user enters "end" terminate the loop
            break
        angle = input("Pick rotation angle between 0 - 359 : \n")
        scale = input("Pick scale positive factor : \n")

        cap.set(cv.CAP_PROP_POS_FRAMES, int(input_frame))
        _, frame = cap.read() # capture the desired frame

        cv.namedWindow("Display window", cv.WINDOW_AUTOSIZE)
        cv.moveWindow("Display window",0,0)
        cv.imshow("Display window", frame)

        cv.setMouseCallback('Display window', mouse_click)
        #### 3D box builder ####
        plt.close()
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)

        # Setting the axes properties
        ax.set_xlim3d([0.0, w])
        ax.set_xlabel('X')

        ax.set_ylim3d([0.0, n])
        ax.set_ylabel('Frame')

        ax.set_zlim3d([0.0, h])
        ax.set_zlabel('Y')

        ax.set_title('3D Test')
        # get the sorted frames & coordinates arrays
        index_list = creatCordinate(frame_list)
        x_array = index_list['x']
        y_array = index_list['y']
        z_array = index_list['z']

        plt.plot(x_array, z_array, y_array)
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(1300, 100, 640, 545)
        plt.show(block=False)

        cv.waitKey()
        plt.close()
        cv.destroyAllWindows()
        #### 3D box builder end ####

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('Output_Video.avi', fourcc, 20.0, (w, h)) # new video setting
    # arrays for interpolated coordinates & frame
    new_x_array = []
    new_y_array = []
    new_z_array = []
    # get the sorted frames & coordinates arrays
    index_list = creatCordinate(frame_list)
    x_array = index_list['x']
    y_array = index_list['y']
    z_array = index_list['z']
    # get the sorted scale arrays
    scale_list_sorted = sorted(scale_list)
    new_scale = []
    for frame in scale_list_sorted:
        new_scale.append(scale_list[frame])
    # get the sorted rotation angles arrays
    rot_list_sorted = sorted(rotate_list)
    new_rotate = []
    for frame in rot_list_sorted:
        new_rotate.append(rotate_list[frame])

    print("Processing video...")
    # looping over sampled frames by array index
    for i in range(len(z_array)):
        # add to arrays of interpolated coordinates & frame
        new_z_array.append(z_array[i])
        new_y_array.append(y_array[i])
        new_x_array.append(x_array[i])
        cap.set(cv.CAP_PROP_POS_FRAMES, z_array[i])
        _, frame = cap.read() # capture the desired frame
        tmp_scale = new_scale[i]
        tmp_rot = new_rotate[i]
        new_img = cv.resize(img, (int(np.ceil(tmp_scale * img_w)), int(np.ceil(tmp_scale * img_h))),
                            interpolation=cv.INTER_CUBIC) # scaling the sampled frame
        new_img = imutils.rotate_bound(new_img, int(tmp_rot)) # rotating the sampled frame
        frame[y_array[i]:y_array[i] + new_img.shape[0], x_array[i]:x_array[i] + new_img.shape[1]] = new_img
        out.write(frame) # save frame to output video
        if z_array[i] == n-1: # if last frame is reached terminate the loop
            break
        else:
            # loop over frames between two sampled frames
            for j in range(z_array[i] + 1, z_array[i+1]):
                cap.set(cv.CAP_PROP_POS_FRAMES, j)
                _, frame = cap.read() # capture the desired frame
                # get the interpolated coordinates
                x1, y1 = find_xy((x_array[i], y_array[i], z_array[i]), (x_array[i+1], y_array[i+1], z_array[i+1]), j)
                # add to arrays of interpolated coordinates & frame
                new_z_array.append(j)
                new_y_array.append(y1)
                new_x_array.append(x1)
                # get the interpolated scale
                tmp_scale = new_scale[i] + (new_scale[i+1] - new_scale[i])*((j-z_array[i])/(z_array[i+1]-z_array[i]))
                # get the interpolated scale
                tmp_rot = new_rotate[i] + (new_rotate[i+1] - new_rotate[i])*((j-z_array[i])/(z_array[i+1]-z_array[i]))
                new_img = cv.resize(img, (int(np.ceil(tmp_scale * img_w)), int(np.ceil(tmp_scale * img_h))),
                                    interpolation=cv.INTER_CUBIC) # scaling the interpolated frame
                new_img = imutils.rotate_bound(new_img, int(tmp_rot)) # rotating the interpolated frame
                frame[y1:y1 + new_img.shape[0], x1:x1 + new_img.shape[1]] = new_img
                out.write(frame) # save frame to output video
    print("Your video is ready!")

    #### final 3D box builder ####
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)

    # Setting the axes properties
    ax.set_xlim3d([0.0, w])
    ax.set_xlabel('X')

    ax.set_ylim3d([0.0, n])
    ax.set_ylabel('Frame')

    ax.set_zlim3d([0.0, h])
    ax.set_zlabel('Y')

    ax.set_title('3D Test')
    index_list = creatCordinate(frame_list)
    new_x_array = index_list['x']
    new_y_array = index_list['y']
    new_z_array = index_list['z']

    plt.plot(new_x_array, new_z_array, new_y_array)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(1300, 100, 640, 545)
    plt.show(block=False)
    # cv.waitKey()
    # cv.destroyAllWindows()
    #### final 3D box builder ####

    # cap.release()
    out.release()
    #cv.destroyAllWindows()

    # new video showing
    cv.namedWindow("Display video", cv.WINDOW_AUTOSIZE)
    cv.moveWindow("Display video", 0, 0)
    cap = cv.VideoCapture('Output_Video.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        cv.imshow('Display video', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])
