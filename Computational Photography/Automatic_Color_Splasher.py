# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:21:02 2017

@author: Cyrus Bonyadi and Jordan Ramus


Fill algorithm modeled on:
http://pillow-cn.readthedocs.io/zh_CN/latest/_modules/PIL/ImageDraw.html
"""
import numpy as np
import cv2

#Public variable for mouse clicks.
click_coordinates = (0, 0) #I really didn't want to do this but see no other way.


def get_image_from_user():
    """ This function gets an image from a user inputted file name.

    Args:
        None

    Returns:
        An image specified by user input.
    """
    
    img = None
    while img is None: #Make sure we get an image file name.
        img_name = input("Where can I access the image that will be changed?\n")
        img = cv2.imread(img_name)
        
        if img is None:#Notify of an error.
            print("Invalid file name.")
            
    return img



def grayscale(image):
    """ This function returns a 3 channel grayscale version of the input image.

    Args:
        An image that will be made grayscale.

    Returns:
        A 3 channel grayscale image.
    """
    
    img_cpy = image.copy()
    
    for r in range(img_cpy.shape[0]):
        for c in range(img_cpy.shape[1]):
            sum = 0. #Sum of channels to be averaged later.
            for channel in (0, 1, 2): #Pan through the channels of each pixel
                sum += float(img_cpy[r][c][channel])
                            
            #Security check for bounds.
            if sum < 0:
                sum = 0.
            elif sum > 765:
                sum = 765.
                
            #Take the average for the pixel.
            average = np.uint8(sum/3.)
            
            #Make a 3 channel grayscale for the pixel.
            for channel in (0, 1, 2):
                img_cpy[r][c][channel] = average
                
    #Return grayscale image.
    return img_cpy


def build_mask(image, threshold_tuple):
    """ This function builds a green border with red fill mask of an image.

    Args:
        A grayscale image that will be edge detected for a mask.

    Returns:
        A 4 channel mask image.
    """
    
    img = image.copy()
    border = cv2.Canny(img, threshold_tuple[0], threshold_tuple[1])
    
    mask = np.zeros((border.shape[0],border.shape[1], 4),np.uint8)
    
    for r in range(border.shape[0]):
        for c in range(border.shape[1]):
            mask[r][c] = (0, border[r][c], 0, border[r][c])
    
             
    #Return grayscale image.
    return mask


def modify_threshold(threshold_tuple, index, value):
    """ This function changes the thresholds in a min, max tuple, keeping rules.
    
    This function will allow you to input a modifying value to a tuple of
    threshold constraints while ensuring the min <= max.

    Args:
        Threshold tuple of size 2, we will verify. Index of 0 or 1 for min 
        or max value. Value to apply to the indexed threshold

    Returns:
        Threshold tuple with change applied
    """
    (min, max) = threshold_tuple #Data validation
    
    
    if index == 0:
        if min + value <= max and min + value >= 0:
            min = min + value
    else :
        if max + value <= 255 and max + value >= min:
            max = max + value
        
    
    return (min, max)

def swap(mask):
    """ This function swaps fill sections of the mask.
    
    This function swaps all fill sections ofa red fill green border mask.

    Args:
        A mask of the border.

    Returns:
        The mask of the border with the fills swapped.
    """
    (rows, cols) = (mask.shape[0], mask.shape[1])
    mask = mask.copy() #Dereference
    
    for r in range(rows):
        for c in range(cols): #All pixels in the image
            if mask[r][c][1] == 255:#Make sure this isn't a border.
                mask[r][c] = (0, 255, 0, 255)
            elif mask[r][c][2] == 255:#See if this is a fill section.
                #Make this an empty section.
                mask[r][c] = (0, 0, 0, 0)
            else:#This must be an empty section.
                mask[r][c] = (0, 0, 255, 100) #Set the alpha to 100: semitransparent
                        
    
    return mask


def dilate(mask):
    """ This function dilates the borders of the mask.
    
    This function enlarges the borders of the mask by 1 pixel in cardinal
    directions.

    Args:
        A mask of the border.

    Returns:
        The mask of the border with the fills swapped.
    """
    (rows, cols) = (mask.shape[0], mask.shape[1])
    new_mask = mask.copy() #Dereference
    
    for r in range(rows):
        for c in range(cols): #All pixels in the image
            if mask[r][c][1] != 255: #Make sure this isn't already a border
                hasNeighbor = False
                for (i, j) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    #Check to ensure we're within bounds.
                    if r+i > 0 and r+i < rows and c+i > 0 and c+i < cols:
                        if mask[r+i][c+i][1] == 255:
                            hasNeighbor = True
                        
                
                if hasNeighbor:
                    new_mask[r][c][1] = 255
                    new_mask[r][c][2] = 0
                    new_mask[r][c][3] = 255
    
    return new_mask

def bridge(mask):
    """ This function bridges the borders of the mask.
    
    This function bridges all small holes in the border of a mask.

    Args:
        A mask of the border.

    Returns:
        The mask of the border with the fills swapped.
    """
    (rows, cols) = (mask.shape[0], mask.shape[1])
    mask = mask.copy() #Dereference
    
    for r in range(rows-1)[1:]:
        for c in range(cols-1)[1:]: #All except the first and last rows
            spans = False
            
            #Check the neighbors to see if this pixel spans a gap in the border
            if mask[r+1][c+1][1] == 255 and mask[r-1][c-1][1] == 255:
                spans = True
            elif mask[r+1][c][1] == 255 and mask[r-1][c][1] == 255:
                spans = True
            elif mask[r][c+1][1] == 255 and mask[r][c-1][1] == 255:
                spans = True
            elif mask[r+1][c-1][1] == 255 and mask[r-1][c+1][1] == 255:
                spans = True
            
            if spans and mask[r][c][1] != 255:
                mask[r][c][1] = 255
                mask[r][c][2] = 0
                mask[r][c][3] = 255
    
    return mask


def bgra_fill_zone(image, coordinates, new_value, border_value):
    """ This function fills within specified borders of an image
    
    Starting from a seed pixel, and boundary set is created.  On each pass,
    the boundary set is cleared and loaded with the next layer of boundarys.

    Args:
        An image, coordinates for a seed pixel, new pixel value, and a border
        value.

    Returns:
        An image with the zone of the click filled.
    """
    #Grab a copy of the image.
    filled_img = image.copy()
    
    #grab limits of the image.
    (rows, cols) = (image.shape[0], image.shape[1])
    
    #grab x and y coordinates 
    (x, y) = coordinates
    
    
    #If the current value of the pixel is correct, stop.
    if np.array_equal(filled_img[x][y], new_value):
        return filled_img
    
    #If the current value of the pixel is the border, stop.
    elif np.array_equal(filled_img[x][y], border_value):
        return filled_img
    
    #Else preload the boundary set and continue.
    else:
        boundary = [(x, y)]
        #While boundary set has values
        while boundary:
            new_boundary = []
            #Check neighbors of values and change values if applicable.
            #Also, add valid neighbors to next boundary set.
            for (x, y) in boundary:
                for(r, c) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                    if r < rows and r >= 0 and c < cols and c >= 0:
                        cur_pxl = filled_img[r][c]
                        if not np.array_equal(cur_pxl, border_value):
                            if not np.array_equal(cur_pxl, new_value):
                                filled_img[r][c] = new_value
                                boundary.append((r, c))
            
            boundary = new_boundary
        
        
        #return the new image.
        return filled_img
    

def fill_mask(mask, coordinates):
    """ This function fills within the borders of the mask.
    
    If the pixel clicked is a fill mask, then it empties the filled zone.
    If the pixel clicked is an empty mask, then it fills the empy zone.

    Args:
        A mask of the border.

    Returns:
        The mask of the border with the fills swapped.
    """
    (x, y) = coordinates
    filled_mask = mask.copy()
    border = np.array([0, 255, 0, 255]) #Green border
    red = np.array([0, 0, 255, 100])
    empty = np.array([0, 0, 0, 0])
    
    
    
    if mask[x][y][1] == 255:
        filled_mask = bgra_fill_zone(mask, coordinates, empty, empty)
    elif mask[x][y][2] == 0:
        filled_mask = bgra_fill_zone(mask, coordinates, red, border)
    elif mask[x][y][2] == 255:
        filled_mask = bgra_fill_zone(mask, coordinates, empty, border)
        
    
    new_mask = filled_mask.copy()
    
    
    return filled_mask


def overlay_mask(bg_tuple, bg_index):
    """ This function overlays a mask on top of the background image.
    
    Args:
        An image tuple with the background options, then a border image.  
        Then a 0 or 1 to specify our background image.

    Returns:
        The border mask overlayed on whichever is specified as the background.
    """
    
    background = bg_tuple[bg_index].copy()
    mask = bg_tuple[2].copy()
    
    result = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    
    #Go through the array
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            #Assign weights to each image based on the alpha values of the mask
            mask_ratio = mask[r][c][3]/255.
            bg_ratio = 1. - mask_ratio
            
            #Take input from each image based on the alpha values of the mask.
            trimmed_mask = mask[:,:,:-1]
            mask_input = mask_ratio*trimmed_mask[r][c]
            bg_input = bg_ratio*background[r][c]
            
            #Output the result of the inputs into a result image.
            result[r][c] = np.uint8(mask_input + bg_input)

    #return the result image
    return result


def finalize(image_tuple, bg_index, cover_index):
    """ This function fills within the borders of the mask.
    
    If the pixel clicked is a fill mask, then it empties the filled zone.
    If the pixel clicked is an empty mask, then it fills the empy zone.

    Args:
        An image tuple with the image options, then a border image.  
        Then an index of our background followed by an index of our foreground.

    Returns:
        The mask applied to our our images with a top and bottom.
    """
    #Get the background and foreground images.
    background = image_tuple[bg_index].copy()
    foreground = image_tuple[cover_index].copy()
    
    #Prepare our result
    result = np.zeros((background.shape[0], background.shape[1], 3), np.uint8) #Empty 3 channel array.
    
    #Go through the image
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            
            #Apply the mask absolutely
            if image_tuple[2][r][c][3] > 0:
                result[r][c] = foreground[r][c]
            else:
                result[r][c] = background[r][c]
    
    return result
  
              
def publish(image_tuple, bg_index, cover_index):
    """ This function publishes a finalized image.
    
    This function calls finalize and publishes the output to a file.
    If you do not enter a file type, it specifies a jpg.

    Args:
        An image tuple with the image options, then a border image.  
        Then an index of our background followed by an index of our foreground.

    Returns:
        The finalized image, just in case.
    """
    publish_image = finalize(image_tuple, bg_index, cover_index)
    
    output_name = input("What would you like to name the output file?\n")
    
    if "." not in output_name:
        output_name += ".jpg"
        
    cv2.imwrite(output_name, publish_image)
    
    return publish_image
    

def click_sub_handler(event, y, x, flags, param):
    """ This function handles a user clicking.
    
    This function is called by setMouseCallback and returns the position 
    of the original click.

    Args:
        The event, coordinates of the event, and necessary flags and params.

    Returns:
        A tuple of x and y coordinates.
    """
	# grab references to the global variables
    global click_coordinates
    
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("Selected for fill:",x, y)
        click_coordinates = (x, y)
        
    return
        


def user_relay(bg_tuple, state, bg_choice, legend):
    """ This function overlays the two images and records user input.
    
    What input is relayed varies based on the state that is passed in.

    Args:
        A tuple containing a gs image, a color image, and a border mask.
        A state of interaction.
        An index of the background image.
        A Bool of legend on or off.

    Returns:
        Returns the output from the user.
    """
    (gs_img, img, border_img) = bg_tuple 
    
    if state == "preview":
        image = finalize(bg_tuple, bg_choice, ((bg_choice + 1)%2))
    else:
        image = overlay_mask(bg_tuple, bg_choice)
    
        
    #Display the image
    cv2.imshow(state, image)
    
    #Print the legend into the console
    print(legend)
    
    #Handle a click if we're in fill mode
    if state == "fill":
        cv2.setMouseCallback(state, click_sub_handler)
    
    #Take a key press
    output = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    #return key pressed.
    return output


def fill_handler(bg_tuple, bg_choice):
    """ This function handles the fill interaction pane of our image.
    Input   | Response:
    O       | Fill Clicked Zone (After a double click)
    S       | Swap Grayscale and Color Background
    G       | Swap Grayscale and Color Zones
    E       | Go to Edit State
    P       | Go to Preview State
    X       | Exit
        

    Args:
        A tuple of variations of an image: a grayscale, the original,
        and an image of its borders.

    Returns:
        Our next state and the border image as it currently stands.
    """
    global click_coordinates
    #This is for data validation.
    (gs_img, img, border_img) = bg_tuple
    border_img = border_img.copy() #Dereference
    state = "fill"
    
    #Pick the background
    bg_img = bg_tuple[bg_choice]    
    
    #create the legend
    legend = "FILL LEGEND:\n\n"
    legend += "Input   | Response:\n"
    legend += "O       | Fill Clicked Zone (After a double click)\n"
    legend += "S       | Swap Grayscale and Color Background\n"
    legend += "G       | Swap Grayscale and Color Zones\n"
    legend += "E       | Go to Edit State\n"
    legend += "P       | Go to Preview State\n"
    legend += "X       | Exit\n"
    legend += "\n\n"
    
    #Fetch user response
    response = user_relay(bg_tuple, state, bg_choice, legend)
    
    #accept a click to fill an area
    if response == ord("o"):
        border_img = fill_mask(bg_tuple[2], click_coordinates)
    
    #swap the background
    elif response == ord("s"):
        bg_choice = (bg_choice + 1) % 2
    
    #swap the grayscale or regular colors
    elif response == ord("g"):
        border_img = swap(border_img)
    
    #go to the edit state
    elif response == ord("e"):
        state = "edit"
    
    #go to the preview state
    elif response == ord("p"):
        state = "preview"
        
    #quit
    elif response == ord("x"):
        state = "end"
    
    return (state, border_img, bg_choice)



def edit_handler(bg_tuple, bg_choice, threshold_tuple):
    """ This function handles the edit interaction pane of our image.
    Input   | Response:
    D       | Dilate Border
    B       | Bridge Border
    S       | Swap Grayscale and Color Background
    G       | Swap Grayscale and Color Zones
    1       | Increase Min_Threshold for Canny Edge by 15
    2       | Decrease Min_Threshold for Canny Edge by 15
    3       | Increase Max_Threshold for Canny Edge by 15
    4       | Decrease Max_Threshold for Canny Edge by 15
    F       | Return to Fill State
    X       | Exit
        

    Args:
        A tuple of variations of an image: a grayscale, the original,
        and an image of its borders.

    Returns:
        Our next state and the border image as it currently stands.
    """
    #This is for data validation.
    (gs_img, img, border_img) = bg_tuple
    border_img = border_img.copy() #Dereference
    state = "edit"
    
    #Pick the background
    bg_img = bg_tuple[bg_choice]
    
    #Create the legend
    legend = "EDIT LEGEND:\n\n"
    legend += "Input   | Response:\n"
    legend += "D       | Dilate Border\n"
    legend += "B       | Bridge Border\n"
    legend += "S       | Swap Grayscale and Color Background\n"
    legend += "G       | Swap Grayscale and Color Zones\n"
    legend += "1       | Increase Min_Threshold for Canny Edge by 15\n"
    legend += "2       | Decrease Min_Threshold for Canny Edge by 15\n"
    legend += "3       | Increase Max_Threshold for Canny Edge by 15\n"
    legend += "4       | Decrease Max_Threshold for Canny Edge by 15\n"
    legend += "F       | Return to Fill State\n"
    legend += "X       | Exit\n"
    legend += "\n"
    legend += "Current Min and Max Canny Tresholds:\n"
    legend += str(threshold_tuple)
    legend += "\n\n\n"
    
    
    #Fetch user input
    response = user_relay(bg_tuple, state, bg_choice, legend)
    
    #dilate the border
    if response == ord("d"):
        border_img = dilate(border_img)
 
    #bridge the border
    elif response == ord("b"):
        border_img = bridge(border_img)
        
    #edit the canny size, then accept or reject changes
    elif response in (ord("1"), ord("2"), ord("3"), ord("4")):
        if response == ord("1"):
            threshold_tuple = modify_threshold(threshold_tuple, 0, 15)
        elif response == ord("2"):
            threshold_tuple = modify_threshold(threshold_tuple, 0, -15)
        elif response == ord("3"):
            threshold_tuple = modify_threshold(threshold_tuple, 1, 15)
        elif response == ord("4"):
            threshold_tuple = modify_threshold(threshold_tuple, 1, -15)
        border_img = build_mask(img, threshold_tuple)
        
    #swap the background
    elif response == ord("s"):
        bg_choice = (bg_choice + 1) % 2
    
    #swap the grayscale or regular colors
    elif response == ord("g"):
        border_img = swap(border_img)
    
    #return to the fill state
    elif response == ord("f"):
        state = "fill"
        
    #quit
    elif response == ord("x"):
        state = "end"
    
    return (state, border_img, bg_choice, threshold_tuple)


def preview_handler(bg_tuple, bg_choice):
    """ This function handles the preview interaction pane of our image.
    Input   | Response:
    W       | Write Image
    S       | Swap Grayscale and Color
    F       | Return to Fill State
    X       | Exit
                
    Args:
        A tuple of variations of an image: a grayscale, the original,
        and an image of its borders.

    Returns:
        Our next state and the border image as it currently stands.
    """
    #This is for data validation.
    (gs_img, img, border_img) = bg_tuple
    border_img = border_img.copy() #Dereference
    state = "preview"
    
    #Pick the background.
    bg_img = bg_tuple[bg_choice]
    
    
    #Create the legend
    legend = "PREVIEW LEGEND:\n\n"
    legend += "Input   | Response:\n"
    legend += "W       | Write Image\n"
    legend += "S       | Swap Grayscale and Color\n"
    legend += "F       | Return to Fill State\n"
    legend += "X       | Exit\n"
    legend += "\n\n"
    
    #Fetch user input.
    response = user_relay(bg_tuple, state, bg_choice, legend)
    
    #enter name and write the image to a file
    if response == ord("w"):
        publish(bg_tuple, bg_choice, (bg_choice+1)%2)
    
    #swap the grayscale or regular colors
    elif response == ord("s"):
        bg_choice = (bg_choice + 1) % 2
    
    #return to the fill state
    elif response == ord("f"):
        state = "fill"
        
    #quit
    elif response == ord("x"):
        state = "end"
    
    return (state, border_img, bg_choice)

          
      
def display_controller(image):
    """ This function controls our handlers to interact with an input image.

    Args:
        An image that will be handled.

    Returns:
        Nothing.
    """
    
    
    #Public variables
    min_threshold = 120 #Min threshold for edge detection
    max_threshold = 210 #Max threshold for edge detection
    
    threshold_tuple = (min_threshold, max_threshold)
    
    bg_choice = 0
    
    gs_img = grayscale(image)
    
    border_img = build_mask(image, threshold_tuple)
    
    state = "fill" #our start state is fill.
    
    while True:
        img_tuple = (gs_img, image, border_img) #Tuple of our working images
        if state is "fill":
            (state, border_img, bg_choice) = fill_handler(img_tuple, bg_choice)
        elif state is "edit":
            (state, border_img, bg_choice, threshold_tuple) = edit_handler(img_tuple, bg_choice, threshold_tuple)
        elif state is "preview":
            (state, border_img, bg_choice) = preview_handler(img_tuple, bg_choice)
        else:
            break
 
       
def main():#Main
    #Initialize
    image = get_image_from_user()
    display_controller(image)
        
        
if __name__ == "__main__": main()  
        
        
        
        
        
        
        
        