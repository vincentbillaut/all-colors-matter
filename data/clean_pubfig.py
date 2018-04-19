import os
FOLDER = 'pubfig/'
list_files = sorted(os.listdir(FOLDER))

#remove empty files + missing images pattern
for file in list_files:
    size = os.path.getsize(FOLDER+file)
    if size<5000 or size==241739:
        os.remove(FOLDER+file)
    else:
        try:
            with open(FOLDER+file,"r") as f:
                if "<!doctype" in f.read()[:1000].lower():
                    os.remove(FOLDER + file)
        except:
            print(file)
#Remove a specific file

#Rename the files