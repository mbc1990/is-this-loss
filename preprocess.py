import os
from shutil import copyfile
from keras.preprocessing.image import load_img

def resize_all(in_dir, out_dir, size):
    """
    Resizes all the images in the directory

    size (x, y) tuple of new size
    """
    fnames = os.listdir(in_dir)
    for fname in fnames:
        img = load_img(in_dir+fname)
        img = img.resize(size)
        try: 
            print fname
            img.save(out_dir+fname)
        except:
            # There are some missing file extensions in the validation set
            img.save(out_dir+fname+".jpg")

def main():
    # Create all the output directories
    os.makedirs("images_processed")

    os.makedirs("images_processed/train")
    os.makedirs("images_processed/train/loss")
    os.makedirs("images_processed/train/other-memes")

    os.makedirs("images_processed/validation")
    os.makedirs("images_processed/validation/loss")
    os.makedirs("images_processed/validation/other-memes")
    
    loss = os.listdir("images_unprocessed/train/loss/")
    other = os.listdir("images_unprocessed/train/other-memes/")
    
    # Used to enfore equal class sizes across input data
    # This makes it easier to judge output performance since you can expect
    # a random accuracy to be 0.50
    class_size = 400
    
    # Split the data in half for training/validation
    num_loss = len(loss)
    for i in range(num_loss):
        if i < class_size / 2:
            copyfile("images_unprocessed/train/loss/" + loss[i],
                     "images_processed/train/loss/" + loss[i])
        else: 
            copyfile("images_unprocessed/train/loss/" + loss[i],
                     "images_processed/validation/loss/" + loss[i])
        if i == class_size:
            break

    num_other = len(other)
    for i in range(num_other):
        if i < class_size / 2:
            copyfile("images_unprocessed/train/other-memes/" + other[i],
                     "images_processed/train/other-memes/" + other[i] + ".jpg")
        else: 
            copyfile("images_unprocessed/train/other-memes/" + other[i],
                     "images_processed/validation/other-memes/" + other[i] + ".jpg")
        if i == class_size:
            break
    

    # Resize every image
    dirs = ["images_processed/train/loss/",
            "images_processed/train/other-memes/",
            "images_processed/validation/loss/",
            "images_processed/validation/other-memes/"]

    output_size = (150, 150)
    for directory in dirs:
        resize_all(directory, directory, output_size)
     

if __name__ == "__main__":
    main()
