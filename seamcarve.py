import cv2
import numpy as np

import sys
import os
import logging
import sys
from datetime import datetime

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S')

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0, 0)

def x_gradient(img):
    '''
    Sobel filter: used for computing the gradient, more resistant to noise
    (img, img's depth, compute dx?, computer dy?, kernel size, ...)
    '''
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)


def y_gradient(img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

# computer energy function
def energy(img):
    blurred = gaussian_blur(img)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    dx = x_gradient(gray)
    dy = y_gradient(gray)
    # final energy = x_grad + y_grad
    return cv2.add(np.absolute(dx), np.absolute(dy))

def cumulative_energies_vertical(energy, protected, mask):
    height, width = energy.shape[:2]
    energies = np.zeros((height, width))
    '''
    For vertical seam:
        M(i, j) = e(i, j) + min(M(i-1, j-1), M(i-1, j), M(i-1, j+1))
    '''
    # for protected: make mask part's energy very high
    if protected:
        energy[mask>0] += 1000
    
    for i in range(1, height):
        for j in range(width):
            left = energies[i - 1, j - 1] if j - 1 >= 0 else 1e6
            middle = energies[i - 1, j]
            right = energies[i - 1, j + 1] if j + 1 < width else 1e6

            energies[i, j] = energy[i, j] + min(left, middle, right)

    return energies

def cumulative_energies_horizontal(energy, protected, mask):
    height, width = energy.shape[:2]
    energies = np.zeros((height, width))
    '''
    For horizontal seam:
        M(i, j) = e(i, j) + min(M(i-1, j-1), M(i, j-1), M(i+1, j-1))
    '''
    # for protected: make mask part's energy very high
    if protected:
        energy[mask>0] += 1000
    
    for j in range(1, width):
        for i in range(height):
            top = energies[i - 1, j - 1] if i - 1 >= 0 else 1e6
            middle = energies[i, j - 1]
            bottom = energies[i + 1, j - 1] if i + 1 < height else 1e6

            energies[i, j] = energy[i, j] + min(top, middle, bottom)

    return energies

def horizontal_seam(energies):
    height, width = energies.shape[:2]
    previous = 0 # the index for seam position now
    seam = []
    # trace the seam back by selecting the minimin previous col pixel
    for i in range(width - 1, -1, -1):
        col = energies[:, i]
        # for last col -> choose the min 'row' as seam end
        if i == width - 1:
            previous = np.argmin(col)

        else:
            top = col[previous - 1] if previous - 1 >= 0 else 1e6
            middle = col[previous]
            bottom = col[previous + 1] if previous + 1 < height else 1e6
            # ex: if select 'top', then previous should -= 1, but argmin return '0' -> minus extra 1
            previous = previous + np.argmin([top, middle, bottom]) - 1

        seam.append([i, previous])

    return seam

def vertical_seam(energies):
    height, width = energies.shape[:2]
    previous = 0
    seam = []
    # trace the seam back by selecting the minimin previous row pixel
    for i in range(height - 1, -1, -1):
        row = energies[i, :]
        # for last row -> choose the min 'col' as seam end
        if i == height - 1:
            previous = np.argmin(row)
            seam.append([previous, i])
        else:
            left = row[previous - 1] if previous - 1 >= 0 else 1e6
            middle = row[previous]
            right = row[previous + 1] if previous + 1 < width else 1e6

            previous = previous + np.argmin([left, middle, right]) - 1
            seam.append([previous, i])

    return seam

def draw_seam(img, seam, interactive=False):
    # (img that lines draw on, vertex coordinate, close or not, line color)
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 0))
    cv2.imshow('seam', img)
    cv2.waitKey(1)

    if not interactive:
        cv2.destroyAllWindows()

def remove_horizontal_seam(img, seam):
    height, width, bands = img.shape
    # as the output after remove min horizontal seam -> new height -= 1 
    removed = np.zeros((height - 1, width, bands), np.uint8)
    # originally seam's index starts from last column, so read the reverse sequence
    for x, y in reversed(seam):
        removed[0:y, x] = img[0:y, x]
        removed[y:height - 1, x] = img[y + 1:height, x]

    return removed

def remove_vertical_seam(img, seam):
    # as the output after remove min vertical seam -> new width -= 1 
    height, width, bands = img.shape
    removed = np.zeros((height, width - 1, bands), np.uint8)
    # originally seam's index starts from last row, so read the reverse sequence
    for x, y in reversed(seam):
        removed[y, 0:x] = img[y, 0:x]
        removed[y, x:width - 1] = img[y, x + 1:width]

    return removed

# Extra add
def remove_horizontal_mask(img, seam):
    height, width = img.shape
    # as the output after remove min horizontal seam -> new height -= 1 
    removed = np.zeros((height - 1, width), np.uint8)
    # originally seam's index starts from last column, so read the reverse sequence
    for x, y in reversed(seam):
        removed[0:y, x] = img[0:y, x]
        removed[y:height - 1, x] = img[y + 1:height, x]

    return removed
# Extra add
def remove_vertical_mask(img, seam):
    # as the output after remove min vertical seam -> new width -= 1 
    height, width = img.shape
    removed = np.zeros((height, width - 1), np.uint8)
    # originally seam's index starts from last row, so read the reverse sequence
    for x, y in reversed(seam):
        removed[y, 0:x] = img[y, 0:x]
        removed[y, x:width - 1] = img[y, x + 1:width]

    return removed

# Extra add
def insert_horizontal_seam(img, seam):
    height, width, ch = img.shape
    output = np.zeros((height+1, width, ch), np.uint8)
    for x, y in reversed(seam):
        for c in range(ch):
            if y==0:
                new = np.average(img[y:y+2, x, c])
                output[y, x, c] = img[y, x, c]
                output[y+1, x, c] = new
                output[y+2:, x, c] = img[y+1: , x, c]
            else:
                new = np.average(img[y-1:y+1, x, c])
                output[:y, x, c] = img[:y, x, c]
                output[y, x, c] = new
                output[y+1: , x, c] = img[y: , x, c]
    return output
    
# Extra add
def insert_vertical_seam(img, seam):
    height, width, ch = img.shape
    output = np.zeros((height, width+1, ch), np.uint8)
    for x, y in reversed(seam):
        for c in range(ch):
            if x==0:
                new = np.average(img[y, x:x+2, c])
                output[y, x, c] = img[y, x, c]
                output[y, x+1, c] = new
                output[y, x+2:, c] = img[y, x+1: , c]
            else:
                new = np.average(img[y, x-1:x+1, c])
                output[y, :x, c] = img[y, :x, c]
                output[y, x, c] = new
                output[y, x+1: , c] = img[y, x: , c]
    return output

# Extra add
def seam_update_horizontal(seam_list, cur_seam):
    output = []
    for seam in seam_list:
        for x, y in reversed(cur_seam):
            if seam[x][1] >= y:
                seam[x][1] += 2

        output.append(seam)
    return output

# Extra add
def seam_update_vertical(seam_list, cur_seam):
    output = []
    for seam in seam_list:
        for x, y in reversed(cur_seam):
            if seam[y][0] >= x:
                seam[y][0] += 2

        output.append(seam)
    return output

    
def window_callback(event, x, y, flags, param):
    """
    Mouse callback function.
    Modify: press the left button(at random position)
            move to the position that user think the area is OK for new size
            then release the left button, monitor will show the new size
    """
    global mx, my
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = x, y
        #print ('Clicked {} x {}.'.format(mx, my))
    if event == cv2.EVENT_LBUTTONUP:
        mx, my = abs(x - mx), abs(y - my)
        print('Now choose %d * %d as new size' %(mx, my))
        print('If OK, then press any key to proceed')

def window_callback2(event, x, y, flags, param):
    
    global protected_top_x, protected_top_y, protected_bottom_x, protected_bottom_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        protected_top_x, protected_top_y = x, y
        protected_bottom_x = -1
        protected_bottom_y = -1
        #print ('Clicked {} x {}.'.format(mx, my))
    if event == cv2.EVENT_LBUTTONUP:
        protected_bottom_x, protected_bottom_y = x, y
        print('Now choose (%d, %d) to (%d, %d) as mask' %(protected_top_x, protected_top_y, protected_bottom_x, protected_bottom_y))
        print('If OK, then press ''Y'' to proceed')
        print('If give up selecting protected mask, then press ''N''')

def resize(img, width=None, height=None, interactive=False):
    result = np.copy(img)

    img_height, img_width = img.shape[:2]
    
    protected = False

    if interactive:
        global mx, my
        mx, my = img_width, img_height
        global protected_top_x, protected_top_y, protected_bottom_x, protected_bottom_y
        protected_top_x=-1 
        protected_top_y=-1
        protected_bottom_x=-1
        protected_bottom_y=-1
        
        cv2.namedWindow('seam', cv2.WINDOW_AUTOSIZE)
        # set custom mouse event: (name, func it call when mouse event appears, param passed to func)
        cv2.setMouseCallback('seam', window_callback, img)
        cv2.imshow('seam', result)
        # waitKey(0) -> wait until pressing any key
        # in this case, user can select any position as prefered cutsize and press key if it's OK
        cv2.waitKey(0)
        # choose proected mask or not
        print('Now select protected mask. If not, press ''N'' to proceed')
        cv2.setMouseCallback('seam', window_callback2, img)
        cv2.imshow('seam', result)
        while True:
            result = np.copy(img)
            if protected_bottom_x!=-1:
                cv2.rectangle(result, (protected_top_x,protected_top_y), (protected_bottom_x,protected_bottom_y), (255,0,0), 2);
                cv2.imshow('seam', result)
                
            key = cv2.waitKey(10)
            if key == ord('n'):
                protected_top_x = -1
                protected_bottom_y = -1
                break
            elif key == ord('y'):
                protected = True
                break

        print ('Resizing to {} (width) x {} (height).'.format(mx, my))
        if key != 'N':
            print ('Protected mask position: ({}*{}) - ({}*{})'.format(protected_top_x, protected_top_y, protected_bottom_x, protected_bottom_y))

    
    result = img
    # create mask: 1 for object, 0 for none
    
    mask = np.zeros((img_height, img_width))
    if protected == True:
        mask[protected_top_x-1:protected_bottom_x, protected_top_y-1:protected_bottom_y] = 1
    
    if interactive:
        cv2.imshow('seam', result)
    
    if height is None:
        height = my

    if width is None:
        width = mx
    # now only support 'downsampling'
    #dy = img_height - height if img_height - height > 0 else 0
    # dx = img_width - width if img_width - width > 0 else 0
    dy = img_height - height
    dx = img_width - width
    # first remove the horizontal seam
    if dy >= 0:
        for i in range(dy):
            energies = cumulative_energies_horizontal(energy(result), protected, mask)
            seam = horizontal_seam(energies)
            if interactive:
                draw_seam(result, seam, interactive=interactive)
            result = remove_horizontal_seam(result, seam)
            mask = remove_horizontal_mask(mask, seam)
    # enlarge the image
    else:
        dy *= -1
        tmp_img = img
        delete_seam = []
        
        for i in range(dy):
            energies = cumulative_energies_horizontal(energy(result), protected, mask)
            seam = horizontal_seam(energies)
            delete_seam.append(seam)
            result = remove_horizontal_seam(result, seam)
            mask = remove_horizontal_mask(mask, seam)
            
        result = tmp_img
        num_row = len(delete_seam)
        for n in range(num_row):
            seam = delete_seam.pop(0)
            result = insert_horizontal_seam(result, seam)
            if interactive:
                draw_seam(result, seam, interactive=interactive)
            delete_seam = seam_update_horizontal(delete_seam, seam)
    
    # then remove the vertical seam
    if dx >= 0:
        for i in range(dx):
            energies = cumulative_energies_vertical(energy(result), protected, mask)
            seam = vertical_seam(energies)
            if interactive:
                draw_seam(result, seam, interactive=interactive)
            result = remove_vertical_seam(result, seam)
            mask = remove_vertical_mask(mask, seam)
    else:
        dx *= -1
        tmp_img = np.copy(result)
        delete_seam = []
        
        for i in range(dx):
            energies = cumulative_energies_vertical(energy(result), protected, mask)
            seam = vertical_seam(energies)
            delete_seam.append(seam)
            result = remove_vertical_seam(result, seam)
            mask = remove_vertical_mask(mask, seam)
        
        result = tmp_img
        num_col = len(delete_seam)
        for n in range(num_col):
            seam = delete_seam.pop(0)
            result = insert_vertical_seam(result, seam)
            if interactive:
                draw_seam(result, seam, interactive=interactive)
            delete_seam = seam_update_vertical(delete_seam, seam)
        
    # give output image name
    name = sys.argv[1][:-3] + '_result.jpg'
    cv2.imwrite(name, result)
    # show the computing time
    logging.info("Processing took %f sec" % ((datetime.now() - start_time).total_seconds()))
    
    print ('Press any key to close the window.')

    cv2.imshow('seam', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def usage(program_name):
    '''Usage: python {} image [--interactive] [new_width new_height]

    --interactive        After starting the program, click in the window to pick
                         the height and width to resize to. Once you've made
                         your final selection, press any key to start the seam
                         carving process.
    new_width            The width to resize the image to.
    new_height           The height to resize the image to.
    '''
    #print (''.format(program_name))

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    # count the time
    logging.info("Processing image")
    start_time = datetime.now()
    # provide interactive way to user
    if len(sys.argv) == 3 and sys.argv[2] == '--interactive':
        resize(img, interactive=True)
    elif len(sys.argv) == 4:
        resize(img, width=int(sys.argv[2]), height=int(sys.argv[3]))
    else:
        usage(sys.argv[0])
        sys.exit(1)

