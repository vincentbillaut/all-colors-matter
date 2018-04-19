SOURCE_FILE = "pubfig/dev_urls.txt"
DEST_FILE = "pubfig/dev_urls_parsed.txt"
dico_people = dict()
ctr_people = dict()

with open(SOURCE_FILE,"r") as f_s:
	with open(DEST_FILE,"w") as f_d:
		for line in f_s:
			if not(line.startswith("#")):
				person,imagenum,url,rect,md5sum = tuple(line.strip().split("\t"))
				person = person.lower()
				if person not in dico_people:
					dico_people[person] = len(dico_people)
					ctr_people[person] = 0
				image_name = "{}_{}".format(dico_people[person],ctr_people[person])
				ctr_people[person]+=1
				f_d.write(image_name+"\t"+url+"\n")

