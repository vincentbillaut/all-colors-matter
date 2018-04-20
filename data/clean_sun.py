import os

PATH1 = "SUN2012/Images/"

list_dirs = os.listdir(PATH1)

for subdir in list_dirs:
    if os.path.isdir(PATH1+subdir):
        for subdir1 in os.listdir(PATH1+subdir):
            if os.path.isdir(PATH1+subdir+"/"+subdir1):
                for subdir2 in os.listdir(PATH1+subdir+"/"+subdir1):
                    if os.path.isdir(PATH1+subdir+"/"+subdir1+"/"+subdir2):
                        for subdir3 in os.listdir(PATH1+subdir+"/"+subdir1+"/"+subdir2):
                            if os.path.isdir(PATH1 + subdir + "/" + subdir1 + "/" + subdir2+"/"+subdir3):
                                print("Need more")
                            else:
                                os.rename(PATH1 + subdir + "/" + subdir1 + "/" + subdir2+ "/" + subdir3,
                                          PATH1 + subdir1 + "_" + subdir2+"_" + subdir3)
                        os.rmdir(PATH1+subdir+"/"+subdir1+"/"+subdir2)
                    else:
                        os.rename(PATH1+subdir+"/"+subdir1+"/"+subdir2,PATH1+subdir+"_"+subdir1+"_"+subdir2)
                os.rmdir(PATH1+subdir+"/"+subdir1)
            else:
                os.rename(PATH1 + subdir + "/" + subdir1 ,
                          PATH1 + subdir + "_" + subdir1)
        os.rmdir(PATH1+subdir)